#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Distributions/MixtureDistribution.h"
#include <regex>

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

constexpr const auto tracy_color = tracy::Color::DeepPink;

GaussianDistribution::GaussianDistribution() : GaussianDistribution({}, {}, 0, REFERENCE_POINT::CENTER)
{
}

GaussianDistribution::GaussianDistribution(Vector mean, Matrix cov)
  : GaussianDistribution(std::move(mean), std::move(cov), 1, REFERENCE_POINT::CENTER)
{
}

GaussianDistribution::GaussianDistribution(Vector mean, Matrix cov, double weight, REFERENCE_POINT ref)
  : _mean(std::move(mean)), _cov(std::move(cov)), _weight{ weight }, _ref_point{ ref }, _id{ _id_Generator.getID() }
{
}

DISTRIBUTION_TYPE GaussianDistribution::type() const
{
  return DISTRIBUTION_TYPE::GAUSSIAN;
}

DistributionId GaussianDistribution::id() const
{
  return _id;
}

REFERENCE_POINT GaussianDistribution::refPoint() const
{
  return _ref_point;
}

bool GaussianDistribution::isValid() const
{
  ZoneScopedNC("GaussianDistribution::isValid", tracy_color);
  if (not _mean.allFinite())
  {
    LOG_FATAL("Mean of Gaussian Distribution is not finite");
    return false;
  }
  if (not _cov.allFinite())
  {
    LOG_FATAL("Covariance of Gaussian Distribution is not finite");
    return false;
  }
  if (_mean.size() > 0 and _cov.size() > 0 and _mean.rows() != _cov.rows())  // Only check if mean, var is set
  {
    LOG_FATAL("Dimension of Mean and of Covariance of Gaussian Distribution does not fit");
    return false;
  }
  if (_weight < 0)
  {
    LOG_FATAL("Weight of Gaussian Distribution is negative");
    return false;
  }
  if (_cov.size() > 0 and ((_cov - _cov.transpose()).array().abs() > 1e-5).any())
  {
    LOG_FATAL("Covariance is not symmetric");
    return false;
  }
  Eigen::LDLT<Matrix> ldlt(_cov);
  if (_cov.size() > 0 and ldlt.info() == Eigen::NumericalIssue)
  {
    Eigen::VectorXcd eigv = _cov.selfadjointView<Eigen::Lower>().eigenvalues();
    if (eigv.real().minCoeff() < -1e-15)
    {
      LOG_FATAL("negative eigenvalues: " << eigv);
      return false;
    }
  }
  return true;
}

Vector const& GaussianDistribution::mean() const
{
  return _mean;
}

Matrix const& GaussianDistribution::covariance() const
{
  return _cov;
}

std::unique_ptr<BaseDistribution> GaussianDistribution::clone() const
{
  auto dist = std::make_unique<GaussianDistribution>(_mean, _cov, _weight, _ref_point);
  dist->_id = _id;
  dist->_prior_id = _prior_id;
  dist->_misc = _misc;
  return dist;
}

void GaussianDistribution::merge(std::unique_ptr<BaseDistribution> other)
{
  std::vector<std::unique_ptr<BaseDistribution>> in;
  in.emplace_back(std::move(other));
  merge(std::move(in));
}

void GaussianDistribution::merge(std::vector<std::unique_ptr<BaseDistribution>> others)
{
  // simple and maybe inefficient -> make mixture dist, compute mean and variance and set as new values for myself
  std::vector<std::unique_ptr<BaseDistribution>> all_comps;
  if (sumWeights() > 0)
  {
    all_comps.emplace_back(clone());
  }
  for (auto const& other : others)
  {
    if (other->sumWeights() > 0)
    {
      for (auto const& comp : other->dists())
      {
        if (comp->refPoint() != _ref_point && _weight > 0.0)
        {
          LOG_FATAL("Requested to merge Component: " << comp->toString() << "into myself: " << toString()
                                                     << " with different REFERENCE POINT");
          throw std::runtime_error("Requested to merge Component with different reference points");
        }
        if (comp->sumWeights() > 0)
        {
          all_comps.emplace_back(comp->clone());
        }
      }
    }
  }
  MixtureDistribution mixture(std::move(all_comps));
  if (mixture.sumWeights() > 0)
  {
    set(mixture.mean(), mixture.covariance());
    set(mixture.sumWeights());
  }
  _id = _id_Generator.getID();
}

BaseDistribution const& GaussianDistribution::bestComponent() const
{
  return *this;
}

double GaussianDistribution::sumWeights() const
{
  return _weight;
}

void GaussianDistribution::multiplyWeights(double fac)
{
  _weight *= fac;
}

std::string GaussianDistribution::toString(std::string const& prefix) const
{
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
  std::ostringstream mean_ss;
  mean_ss << prefix + "|\tMean: " << mean().format(fmt);
  auto mean_str = mean_ss.str();
  mean_str = std::regex_replace(mean_str, std::regex("\n"), "\n" + prefix + "|\t      ");

  std::ostringstream cov_ss;
  cov_ss << prefix + "|\tCov: " << covariance().format(fmt);
  auto cov_str = cov_ss.str();
  cov_str = std::regex_replace(cov_str, std::regex("\n"), "\n" + prefix + "|\t      ");

  std::ostringstream oss;
  oss << prefix + "Gaussian Distribution\n" + prefix + "|\tID: " + std::to_string(_id.value_) + '\n' + prefix +
             "|\tPrior ID:"
      << _prior_id << '\n'
      << mean_str << '\n'
      << cov_str << '\n'
      << prefix << "|\tWeight: " << sumWeights() << '\n'
      << prefix << "|\tRefPoint: " << to_string(_ref_point) << '\n';
  if (not _misc.empty())
  {
    for (auto const& val : _misc)
    {
      oss << prefix << "|\tMisc given for: " + val.first + '\n';
    }
    if (_misc.contains("ny"))
    {
      oss << prefix << "|\tny: " + std::to_string(any_cast<double>(_misc.at("ny"))) + '\n';
    }
  }
  return oss.str();
}

std::size_t GaussianDistribution::pruneWeight(double weightThreshold)
{
  if (_weight < weightThreshold)
  {
    _mean = {};
    _cov = {};
    _weight = 0;
    return 1;
  }
  return 0;
}

std::size_t GaussianDistribution::pruneVar(double varThreshold)
{
  if (_cov.size() > 0 and _cov.maxCoeff() > varThreshold)
  {
    _mean = {};
    _cov = {};
    _weight = 0;
    return 1;
  }
  return 0;
}

std::size_t GaussianDistribution::truncate(std::size_t maxNComponents)
{
  return 0;
}

std::size_t GaussianDistribution::mergeComponents(double maxDist, Components const& comps)
{
  return 0;
}

void GaussianDistribution::set(Vector mean)
{
  _mean = std::move(mean);
  m_infoVec.reset();
  _id = _id_Generator.getID();
}
void GaussianDistribution::set(Matrix cov)
{
  _cov = std::move(cov);
  m_infoMat.reset();
  _id = _id_Generator.getID();
}

void GaussianDistribution::set(Vector mean, Matrix cov)
{
  _mean = std::move(mean);
  _cov = std::move(cov);
  m_infoMat.reset();
  m_infoVec.reset();
  _id = _id_Generator.getID();
}

void GaussianDistribution::set(double weight)
{
  _weight = weight;
  _id = _id_Generator.getID();
}

void GaussianDistribution::set(ttb::REFERENCE_POINT refPoint)
{
  _ref_point = refPoint;
  _id = _id_Generator.getID();
}

void GaussianDistribution::setPriorId(DistributionId new_id)
{
  _prior_id = new_id;
}

DistributionId GaussianDistribution::priorId() const
{
  return _prior_id;
}

void GaussianDistribution::resetPriorId()
{
  _prior_id = NO_DISTRIBUTION_ID_HISTORY;
}

Matrix const& GaussianDistribution::getInfoMat() const
{
  if (not m_infoMat.has_value())
  {
    m_infoMat = covariance().inverse();
  }
  return m_infoMat.value();
}

Vector const& GaussianDistribution::getInfoVec() const
{
  if (not m_infoVec.has_value())
  {
    m_infoVec = getInfoMat() * mean();
  }
  return m_infoVec.value();
}

double GaussianDistribution::pdf(Vector const& x) const
{
  Vector const res = x - _mean;
  return 1 / std::sqrt((2 * std::numbers::pi * _cov).determinant()) *
         std::exp(-0.5 * res.transpose() * getInfoMat() * res);
}

std::vector<BaseDistribution*> GaussianDistribution::dists()
{
  if (_weight < TTB_EPS)
  {
    return {};
  }
  return { this };
}

std::vector<BaseDistribution const*> GaussianDistribution::dists() const
{
  if (_weight < TTB_EPS)
  {
    return {};
  }
  return { this };
}

}  // namespace ttb
