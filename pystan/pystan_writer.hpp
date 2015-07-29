#ifndef PYSTAN__PYSTAN_WRITER_HPP
#define PYSTAN__PYSTAN_WRITER_HPP

#include <stan/interface_callbacks/writer/csv.hpp>
#include <stan/interface_callbacks/writer/filtered_values.hpp>
#include <stan/interface_callbacks/writer/sum_values.hpp>


namespace pystan {

  class pystan_sample_writer {
  public:
    typedef stan::interface_callbacks::writer::csv CsvWriter;
    typedef stan::interface_callbacks::writer::filtered_values<std::vector<double> > FilteredValuesWriter;
    typedef stan::interface_callbacks::writer::sum_values SumValuesWriter;

    CsvWriter csv_;
    FilteredValuesWriter values_;
    FilteredValuesWriter sampler_values_;
    SumValuesWriter sum_;

    pystan_sample_writer(CsvWriter csv, FilteredValuesWriter values, FilteredValuesWriter sampler_values, SumValuesWriter sum)
      : csv_(csv), values_(values), sampler_values_(sampler_values), sum_(sum) { }

    void operator()(const std::vector<std::string>& x) {
      csv_(x);
      values_(x);
      sampler_values_(x);
      sum_(x);
    }

    template <class T>
    void operator()(const std::vector<T>& x) {
      csv_(x);
      values_(x);
      sampler_values_(x);
      sum_(x);
    }

    void operator()(const std::string x) {
      csv_(x);
      values_(x);
      sampler_values_(x);
      sum_(x);
    }

    void operator()() {
      csv_();
      values_();
      sampler_values_();
      sum_();
    }

    bool is_writing() const {
      return csv_.is_writing() || values_.is_writing() || sampler_values_.is_writing() || sum_.is_writing();
    }
  };

  /**
    @param      N
    @param      M    number of iterations to be saved
    @param      warmup    number of warmup iterations to be saved
   */

  pystan_sample_writer
  sample_writer_factory(std::ostream *o, const std::string prefix,
                          const size_t N, const size_t M, const size_t warmup,
                          const size_t offset,
                          const std::vector<size_t>& qoi_idx) {
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

    stan::interface_callbacks::writer::csv csv(o, prefix);
    stan::interface_callbacks::writer::filtered_values<std::vector<double> > values(N, M, filter);
    stan::interface_callbacks::writer::filtered_values<std::vector<double> > sampler_values(N, M, filter_sampler_values);
    stan::interface_callbacks::writer::sum_values sum(N, warmup);

    return pystan_sample_writer(csv, values, sampler_values, sum);
  }

  stan::interface_callbacks::writer::csv
  diagnostic_writer_factory(std::ostream *o, const std::string prefix) {
    stan::interface_callbacks::writer::csv csv(o, prefix);
    return csv;
  }

}


#endif
