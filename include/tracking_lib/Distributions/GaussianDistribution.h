#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Concepts.h"
#include <regex>
#include <stacktrace>

namespace ttb
{

template <class UDT>
class GaussianDistribution
{
public:
  GaussianDistribution(Vector<UDT> mean, Matrix<UDT> cov) : _mean{ std::move(mean) }, _cov{ std::move(cov) }
  {
  }

  GaussianDistribution(Vector<UDT> mean, Matrix<UDT> cov, REFERENCE_POINT ref)
    : _mean{ std::move(mean) }, _cov{ std::move(cov) }, _ref_point{ ref }
  {
  }

  [[nodiscard]] DistributionId id() const
  {
    return _id;
  }

  [[nodiscard]] REFERENCE_POINT ref_point() const
  {
    return _ref_point;
  }

  [[nodiscard]] REFERENCE_POINT& ref_point()
  {
    return _ref_point;
  }

  [[nodiscard]] bool is_empty() const noexcept
  {
    return _mean.size() == 0;
  }

  [[nodiscard]] bool is_valid() const
  {
    if (_mean.size() == 0 or _cov.size() == 0)
    {
      LOG_FATAL("Dimension 0 in mean or covariance");
      LOG_FATAL(to_string());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
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

    if (_mean.rows() != _cov.rows() or _mean.rows() != _cov.cols())
    {
      LOG_FATAL("Dimension of Mean and of Covariance of Gaussian Distribution does not fit");
      return false;
    }
    if (((_cov - _cov.transpose()).array().abs() > 1e-5).any())
    {
      LOG_FATAL("Covariance is not symmetric");
      return false;
    }
    auto const adjoint = _cov.template selfadjointView<Eigen::Lower>();
    Eigen::LDLT ldlt(adjoint.ldlt());
    if (ldlt.info() == Eigen::NumericalIssue)
    {
      LOG_FATAL("eigen numerical issues");
      LOG_FATAL(std::stacktrace::current());
      LOG_FATAL("determinant: " << _cov.determinant());
      LOG_FATAL(to_string());
      return false;
    }
    if (adjoint.eigenvalues().real().minCoeff() < -1e-7)
    {
      LOG_FATAL("negative eigenvalues: " << adjoint.eigenvalues());
      return false;
    }
    if ((ldlt.vectorD().array() < -1e-7).any())
    {
      LOG_FATAL("negative eigenvalues: " << ldlt.vectorD());
      return false;
    }
    return true;
  }

  [[nodiscard]] Matrix<UDT> const& covariance() const
  {
    return _cov;
  }

  [[nodiscard]] Matrix<UDT>& covariance()
  {
    return _cov;
  }

  [[nodiscard]] Vector<UDT> const& mean() const
  {
    return _mean;
  }

  [[nodiscard]] Vector<UDT>& mean()
  {
    return _mean;
  }

  [[nodiscard]] std::string to_string(std::string const& prefix = "") const
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
        << prefix << "|\tRefPoint: " << ttb::to_string(_ref_point) << '\n';
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

  std::size_t prune_var(double threshold)
  {
    assert(is_valid());
    if (_cov.maxCoeff() > threshold)
    {
      _mean = {};
      _cov = {};
      return 1;
    }
    return 0;
  }

  [[nodiscard]] DistributionId prior_id() const
  {
    return _prior_id;
  }

  [[nodiscard]] DistributionId& prior_id()
  {
    return _prior_id;
  }

  [[nodiscard]] std::map<std::string, std::any>& misc()
  {
    return _misc;
  }

  [[nodiscard]] const std::map<std::string, std::any>& misc() const
  {
    return _misc;
  }

  void give_new_id()
  {
    _id = _dist_id_Generator.getID();
  }

  void pb_legacy_set(Vector<UDT> mean)
  {
    _mean = std::move(mean);
    _id = _dist_id_Generator.getID();
  }

  void pb_legacy_set(Matrix<UDT> cov)
  {
    _cov = std::move(cov);
    _id = _dist_id_Generator.getID();
  }

  void pb_legacy_set(REFERENCE_POINT rp)
  {
    _ref_point = rp;
    _id = _dist_id_Generator.getID();
  }

private:
  Vector<UDT> _mean{};
  Matrix<UDT> _cov{};
  REFERENCE_POINT _ref_point{ REFERENCE_POINT::CENTER };
  DistributionId _id{ _dist_id_Generator.getID() };
  DistributionId _prior_id{ NO_DISTRIBUTION_ID_HISTORY };
  std::map<std::string, std::any> _misc{};
};

static_assert(CPTDistribution<GaussianDistribution<double>>);
static_assert(CPTDistribution<GaussianDistribution<AD>>);

template <class UDT>
[[nodiscard]] inline double kl_divergence(GaussianDistribution<UDT> const& first,
                                          GaussianDistribution<UDT> const& second)
{
  auto const first_llt = first.covariance().llt();
  auto const second_llt = second.covariance().llt();
  Matrix<UDT> const L0 = first_llt.matrixL();
  Matrix<UDT> const M = second_llt.matrixL().solve(L0);
  Vector<UDT> const y = second_llt.matrixL().solve(second.mean() - first.mean());
  double chol_part = 0;
  for (Index i = 0; i < first.mean().rows(); ++i)
  {
    chol_part += std::log(second_llt.matrixL()(i, i) / L0(i, i));
  }
  return 0.5 * (M.array().pow(2).sum() - first.mean().rows() + y.array().pow(2).sum() + 2 * chol_part);
}

template <class UDT>
[[nodiscard]] inline UDT kl_divergence_naive(GaussianDistribution<UDT> const& first,
                                                GaussianDistribution<UDT> const& second)
{
  Vector<UDT> const res = second.mean() - first.mean();
  return 0.5 * ((second.covariance().inverse() * first.covariance()).trace() - first.mean().rows() +
                res.transpose() * second.covariance().inverse() * res +
                std::log(second.covariance().determinant() / first.covariance().determinant()));
}

}  // namespace ttb