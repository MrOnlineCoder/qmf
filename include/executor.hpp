#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <algorithm>
#include <vector>

#include <Eigen/Dense>

typedef uint64_t bignum_t;

class Executor {
public:
    Executor();

    void changeVectorSpaceSize(int size, int* alpha = nullptr);

    bool calculateMonotonicity(std::size_t functionNumber, bool debug = false);

    std::vector<int8_t> useQuickTransformation(std::vector<uint8_t> f, bool inverse);

    const bignum_t getTotalFunctionsCount() const;

   private:
    std::vector<uint8_t> getLogicalFunction(std::size_t functionNumber);

    const bignum_t getMaxSetsCount() const;

    int mVectorSpaceSize;
    int* m_alphaSet;

    Eigen::MatrixXf mTransitionMatrix;
    Eigen::MatrixXf mTransitionMatrixInverse;
};

#endif