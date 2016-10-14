#ifndef PYSTAN__PYSTAN_WRITER_HPP
#define PYSTAN__PYSTAN_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>

namespace pystan {

  class value : public stan::callbacks::writer {
  private:
    std::vector<double> x_;

  public:
    value() { }

    void operator()(const std::string& key, double value) { }

    void operator()(const std::string& key, int value) { }

    void operator()(const std::string& key, const std::string& value) { }

    void operator()(const std::string& key, const double* values,
                    int n_values) { }

    void operator()(const std::string& key, const double* values,
                    int n_rows, int n_cols) { }

    void operator()(const std::vector<std::string>& names) { }

    void operator()(const std::vector<double>& x) {
      x_ = x;
    }

    void operator()(const std::string& message) { }

    void operator()() { }

    const std::vector<double> x() const {
      return x_;
    }
  };

  template <class InternalVector>
  class values : public stan::callbacks::writer {
  private:
    size_t m_;
    size_t N_;
    size_t M_;
    std::vector<InternalVector> x_;

  public:
    values(const size_t N,
           const size_t M)
      : m_(0), N_(N), M_(M) {
      x_.reserve(N_);
      for (size_t n = 0; n < N_; n++)
        x_.push_back(InternalVector(M_));
    }

    explicit values(const std::vector<InternalVector>& x)
      : m_(0), N_(x.size()), M_(0),
        x_(x) {
      if (N_ > 0)
        M_ = x_[0].size();
    }

    void operator()(const std::string& key,
                    double value) { }

    void operator()(const std::string& key,
                    int value) { }

    void operator()(const std::string& key,
                    const std::string& value) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_values) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_rows, int n_cols) { }

    void operator()(const std::vector<std::string>& names) { }

    void operator()(const std::vector<double>& x) {
      if (N_ != x.size())
        throw std::length_error("vector provided does not "
                                "match the parameter length");
      if (m_ == M_)
        throw std::out_of_range("");
      for (size_t n = 0; n < N_; n++)
        x_[n][m_] = x[n];
      m_++;
    }

    void operator()() { }

    void operator()(const std::string& message) { }


    const std::vector<InternalVector>& x() const {
      return x_;
    }
  };

  template <class InternalVector>
  class filtered_values : public stan::callbacks::writer {
  private:
    size_t N_, M_, N_filter_;
    std::vector<size_t> filter_;
    values<InternalVector> values_;
    std::vector<double> tmp;

  public:
    filtered_values(const size_t N,
                    const size_t M,
                    const std::vector<size_t>& filter)
      : N_(N), M_(M), N_filter_(filter.size()), filter_(filter),
        values_(N_filter_, M_), tmp(N_filter_) {
      for (size_t n = 0; n < N_filter_; n++)
        if (filter.at(n) >= N_)
          throw std::out_of_range("filter is looking for "
                                  "elements out of range");
    }

    filtered_values(const size_t N,
                    const std::vector<InternalVector>& x,
                    const std::vector<size_t>& filter)
      : N_(N), M_(0), filter_(filter), N_filter_(filter.size()),
        values_(x), tmp(N_filter_) {
      if (x.size() != filter.size())
        throw std::length_error("filter provided does not "
                                "match dimensions of the storage");
      if (N_filter_ > 0)
        M_ = x[0].size();
      for (size_t n = 0; n < N_filter_; n++)
        if (filter.at(n) >= N_)
          throw std::out_of_range("filter is looking for "
                                  "elements out of range");
    }

    void operator()(const std::string& key,
                    double value) { }

    void operator()(const std::string& key,
                    int value) { }

    void operator()(const std::string& key,
                    const std::string& value) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_values) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_rows, int n_cols) { }

    void operator()(const std::vector<std::string>& names) {
    }

    void operator()(const std::vector<double>& state) {
      if (state.size() != N_)
        throw std::length_error("vector provided does not "
                                "match the parameter length");
      for (size_t n = 0; n < N_filter_; n++)
        tmp[n] = state[filter_[n]];
      values_(tmp);
    }

    void operator()(const std::string& message) { }

    void operator()() { }

    const std::vector<InternalVector>& x() {
      return values_.x();
    }
  };

  class sum_values : public stan::callbacks::writer {
  public:
    explicit sum_values(const size_t N)
      : N_(N), m_(0), skip_(0), sum_(N_, 0.0) { }

    sum_values(const size_t N, const size_t skip)
      : N_(N), m_(0), skip_(skip), sum_(N_, 0.0) { }


    void operator()(const std::string& key,
                    double value) { }

    void operator()(const std::string& key,
                    int value) { }

    void operator()(const std::string& key,
                    const std::string& value) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_values) { }

    void operator()(const std::string& key,
                    const double* values,
                    int n_rows, int n_cols) { }

    /**
     * Do nothing with std::string vector
     *
     * @param names
     */
    void operator()(const std::vector<std::string>& names) { }


    /**
     * Add values to cumulative sum
     *
     * @param x vector of type T
     */
    void operator()(const std::vector<double>& state) {
      if (N_ != state.size())
        throw std::length_error("vector provided does not "
                                "match the parameter length");
      if (m_ >= skip_) {
        for (size_t n = 0; n < N_; n++)
          sum_[n] += state[n];
      }
      m_++;
    }


    /**
     * Do nothing with a string.
     *
     * @param x string to print with prefix in front
     */
    void operator()(const std::string& message) { }

    /**
     * Do nothing
     *
     */
    void operator()() { }

    const std::vector<double>& sum() const {
      return sum_;
    }

    const size_t called() const {
      return m_;
    }

    const size_t recorded() const {
      if (m_ >= skip_)
        return m_ - skip_;
      else
        return 0;
    }

  private:
    size_t N_;
    size_t m_;
    size_t skip_;
    std::vector<double> sum_;
  };

  class comment_writer : public stan::callbacks::writer {
  private:
    stan::callbacks::stream_writer writer_;
  public:
    comment_writer(std::ostream& stream, const std::string& prefix = "")
      : writer_(stream, prefix) {
    }

    void operator()(const std::string& key, double value) {
      writer_(key, value);
    }

    void operator()(const std::string& key, int value) {
      writer_(key, value);
    }

    void operator()(const std::string& key, const std::string& value) {
      writer_(key, value);
    }

    void operator()(const std::string& key, const double* values,
                    int n_values) { }

    void operator()(const std::string& key, const double* values,
                    int n_rows, int n_cols) { }

    void operator()(const std::vector<std::string>& names) { }

    void operator()(const std::vector<double>& x) { }

    void operator()(const std::string& message) {
      writer_(message);
    }

    void operator()() {
      writer_();
    }
  };


  class pystan_sample_writer : public stan::callbacks::writer {
  public:
    stan::callbacks::stream_writer csv_;
    comment_writer comment_writer_;
    filtered_values<std::vector<double> > values_;
    filtered_values<std::vector<double> > sampler_values_;
    sum_values sum_;

    pystan_sample_writer(stan::callbacks::stream_writer csv,
                         comment_writer comment_writer,
                         filtered_values<std::vector<double> > values,
                         filtered_values<std::vector<double> > sampler_values,
                         sum_values sum)
      : csv_(csv), comment_writer_(comment_writer),
        values_(values), sampler_values_(sampler_values), sum_(sum) { }

    /**
     * Writes a key, value pair.
     *
     * @param[in] key A string
     * @param[in] value A double value
     */
    void operator()(const std::string& key, double value) {
      csv_(key, value);
      comment_writer_(key, value);
      values_(key, value);
      sampler_values_(key, value);
      sum_(key, value);
    }

    /**
     * Writes a key, value pair.
     *
     * @param[in] key A string
     * @param[in] value An integer value
     */
    void operator()(const std::string& key, int value) {
      csv_(key, value);
      comment_writer_(key, value);
      values_(key, value);
      sampler_values_(key, value);
      sum_(key, value);
    }

    /**
     * Writes a key, value pair.
     *
     * @param[in] key A string
     * @param[in] value A string
     */
    void operator()(const std::string& key, const std::string& value) {
      csv_(key, value);
      comment_writer_(key, value);
      values_(key, value);
      sampler_values_(key, value);
      sum_(key, value);
    }

    /**
     * Writes a key, value pair.
     *
     * @param[in] key A string
     * @param[in] values A double array, typically used with
     *   contiguous Eigen vectors
     * @param[in] n_values Length of the array
     */
    void operator()(const std::string& key, const double* values,
                    int n_values)  {
      csv_(key, values, n_values);
      comment_writer_(key, values, n_values);
      values_(key, values, n_values);
      sampler_values_(key, values, n_values);
      sum_(key, values, n_values);
    }

    /**
     * Writes a key, value pair.
     *
     * @param[in] key A string
     * @param[in] values A double array assumed to represent a 2d
     *   matrix stored in column major order, typically used with
     *   contiguous Eigen matrices
     * @param[in] n_rows Rows
     * @param[in] n_cols Columns
     */
    void operator()(const std::string& key, const double* values,
                    int n_rows, int n_cols) {
      csv_(key, values, n_rows, n_cols);
      comment_writer_(key, values, n_rows, n_cols);
      values_(key, values, n_rows, n_cols);
      sampler_values_(key, values, n_rows, n_cols);
      sum_(key, values, n_rows, n_cols);
    }

    /**
     * Writes a set of names.
     *
     * @param[in] names Names in a std::vector
     */
    void operator()(const std::vector<std::string>& names) {
      csv_(names);
      comment_writer_(names);
      values_(names);
      sampler_values_(names);
      sum_(names);
    }

    /**
     * Writes a set of values.
     *
     * @param[in] state Values in a std::vector
     */
    void operator()(const std::vector<double>& state) {
      csv_(state);
      comment_writer_(state);
      values_(state);
      sampler_values_(state);
      sum_(state);
    }

    /**
     * Writes a string.
     *
     * @param[in] message A string
     */
    void operator()(const std::string& message) {
      csv_(message);
      comment_writer_(message);
      values_(message);
      sampler_values_(message);
      sum_(message);
    }

    /**
     * Writes blank input.
     */
    void operator()() {
      csv_();
      comment_writer_();
      values_();
      sampler_values_();
      sum_();
    }
  };

  /**
     @param      N
     @param      M    number of iterations to be saved
     @param      warmup    number of warmup iterations to be saved
  */
  pystan_sample_writer*
  sample_writer_factory(std::ostream *csv_fstream,
                        std::ostream& comment_stream,
                        const std::string& prefix,
                        size_t N_sample_names, size_t N_sampler_names,
                        size_t N_constrained_param_names,
                        size_t N_iter_save, size_t warmup,
                        const std::vector<size_t>& qoi_idx) {
    size_t N = N_sample_names + N_sampler_names + N_constrained_param_names;
    size_t offset = N_sample_names + N_sampler_names;

    std::vector<size_t> filter(qoi_idx);
    std::vector<size_t> lp;
    for (size_t n = 0; n < filter.size(); n++)
      if (filter[n] >= N)
        lp.push_back(n);
    for (size_t n = 0; n < filter.size(); n++)
      filter[n] += offset;
    for (size_t n = 0; n < lp.size(); n++)
      filter[lp[n]] = 0;

    std::vector<size_t> filter_sampler_values(offset);
    for (size_t n = 0; n < offset; n++)
      filter_sampler_values[n] = n;

    stan::callbacks::stream_writer csv(*csv_fstream, prefix);
    comment_writer comments(comment_stream, prefix);
    filtered_values<std::vector<double> > values(N, N_iter_save, filter);
    filtered_values<std::vector<double> > sampler_values(N, N_iter_save, filter_sampler_values);
    sum_values sum(N, warmup);

    return new pystan_sample_writer(csv, comments, values, sampler_values, sum);
  }

}
#endif
