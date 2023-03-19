#include <Eigen/KroneckerProduct>
#include <chrono>
#include <executor.hpp>
#include <iostream>

Executor::Executor() {
    changeVectorSpaceSize(2);
}

void Executor::changeVectorSpaceSize(int size) {
    mVectorSpaceSize = size;

    Eigen::MatrixXf mConstant{
        {1, 0},
        {1, 1}};

    mTransitionMatrix = mConstant;

    for (int i = 1; i < mVectorSpaceSize; i++) {
        auto newMatrix = Eigen::kroneckerProduct(mTransitionMatrix, mConstant);
        mTransitionMatrix = newMatrix.eval();
    }

    mTransitionMatrixInverse = mTransitionMatrix.inverse().transpose();

    std::cout
        << "Kf:\n"
        << mTransitionMatrix << std::endl;

    std::cout
        << "Kf_inverse:\n"
        << mTransitionMatrixInverse << std::endl;
}

int quickTransformer(std::vector<uint8_t>& f, int subIndex) {
}

std::vector<uint8_t> Executor::useQuickTransformation(std::vector<uint8_t> f) {
    std::vector<uint8_t> r;
    r.resize(f.size());

    for (int i = 0; i < f.size(); i++) {
        r[i] = f[i];
    }
}

bool Executor::calculateMonotonicity(std::size_t functionNumber, bool debug) {
    auto begin = std::chrono::high_resolution_clock::now();

    auto func = getLogicalFunction(functionNumber);

    if (debug) {
        std::cout << "f = ( ";

        for (int i = 0; i < func.size(); i++) {
            std::cout << (func[i] == 0 ? "0" : "1") << " ";
        }

        std::cout << ")" << std::endl;
    }

    Eigen::VectorXf fVector(getMaxSetsCount());

    for (int i = 0; i < getMaxSetsCount(); i++) {
        fVector[i] = func[i];
    }

    auto fEnergySpectreVector = (mTransitionMatrix * fVector).cwiseProduct(mTransitionMatrixInverse * fVector);

    auto fEnergyValue = fVector.dot(fVector);

    auto end = std::chrono::high_resolution_clock::now();

    if (debug) {
        std::cout << "f energy = " << fEnergyValue << std::endl;
        std::cout << "f energy spectre = ( " << fEnergySpectreVector.transpose() << " )" << std::endl;
        std::cout << "(" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "mcs )" << std::endl;
    }

    for (int i = 0; i < fEnergySpectreVector.size(); i++) {
        if (i == fEnergySpectreVector.size() - 1) {
            if (std::abs(fEnergySpectreVector[i] - fEnergyValue) > 0.01) {
                return false;
            }
        } else {
            if (std::abs(fEnergySpectreVector[i]) > 0.01) {
                return false;
            }
        }
    }

    return true;
}

std::vector<uint8_t> Executor::getLogicalFunction(std::size_t functionNumber) {
    auto funcVectorSize = getMaxSetsCount();

    std::vector<uint8_t> result;
    result.resize(funcVectorSize);

    for (std::size_t i = 0; i < funcVectorSize; ++i) {
        result[funcVectorSize - i - 1] = (functionNumber >> i) & 1;
    }

    return result;
}

const bignum_t Executor::getMaxSetsCount() const {
    return std::pow(2, mVectorSpaceSize);
}

const bignum_t Executor::getTotalFunctionsCount() const {
    return std::pow(2, getMaxSetsCount());
}