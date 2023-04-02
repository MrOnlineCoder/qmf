#include <Eigen/KroneckerProduct>
#include <chrono>
#include <executor.hpp>
#include <iostream>

Eigen::MatrixXf logicalTrueConstantMatrix{
    {1, 0},
    {1, 1}};

Eigen::MatrixXf logicalFalseConstantMatrix{
    {1, 1},
    {0, 1}};

Eigen::MatrixXf getConstantMatrix(bool value) {
    return value ? logicalTrueConstantMatrix : logicalFalseConstantMatrix;
}

Executor::Executor() {
    changeVectorSpaceSize(2);
}

void Executor::changeVectorSpaceSize(int size, int* alpha) {
    mVectorSpaceSize = size;

    m_alphaSet = new int[size];

    if (alpha == nullptr) {
        for (int i = 0; i < size; i++) {
            m_alphaSet[i] = 1;
        }
    } else {
        m_alphaSet = alpha;
    }

    mTransitionMatrix = getConstantMatrix(m_alphaSet[0]);

    for (int i = 1; i < mVectorSpaceSize; i++) {
        auto constant = getConstantMatrix(m_alphaSet);
        auto newMatrix = Eigen::kroneckerProduct(mTransitionMatrix, constant);
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

void quickTransformer(std::vector<int8_t>& f, int subIndex, int n) {
    int half = std::pow(2, n) / 2;

    int npower = std::pow(2, n);
    std::cout << " half = " << half << " npower = " << npower << " subIndex = " << subIndex << " n = " << n << std::endl;

    std::vector<int8_t> fo = f;

    for (int i = 0; i < f.size(); i++) {
        if (i < half) {
            if (subIndex == 0) {
                f[i] = fo[2 * i] + fo[2 * i - npower + 1];
            } else {
                f[i] = fo[2 * i];
            }
        } else {
            if (subIndex == 0) {
                f[i] = fo[2 * i - npower + 1];
            } else {
                f[i] = fo[2 * i - npower] + fo[2 * i - npower + 1];
            }
        }
    }
}

std::vector<int8_t> Executor::useQuickTransformation(std::vector<uint8_t> f) {
    std::vector<int8_t> r;
    r.resize(f.size());

    for (int i = 0; i < f.size(); i++) {
        r[i] = f[i];
    }

    for (int i = 0; i < mVectorSpaceSize; i++) {
        int subIndex = m_alphaSet[mVectorSpaceSize - i - 1];
        quickTransformer(r, subIndex, mVectorSpaceSize);
    }

    return r;
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

    auto fKfVector = mTransitionMatrix * fVector;
    auto fEnergySpectreVector = (mTransitionMatrix * fVector).cwiseProduct(mTransitionMatrixInverse * fVector);

    auto qbegin = std::chrono::high_resolution_clock::now();
    auto fQuickTransformed = useQuickTransformation(func);
    auto qend = std::chrono::high_resolution_clock::now();
    auto qtime = std::chrono::duration_cast<std::chrono::microseconds>(qend - qbegin).count();

    auto fEnergyValue = fVector.dot(fVector);

    auto end = std::chrono::high_resolution_clock::now();

    if (debug) {
        std::cout << "f energy = " << fEnergyValue << std::endl;
        std::cout << "f energy spectre = ( " << fEnergySpectreVector.transpose() << " )" << std::endl;
        std::cout << "matrix time => (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "mcs )" << std::endl;
        std::cout << "quick transform vector matrix = " << fKfVector.transpose() << "\n"; 
        std::cout << "quick transform = (";
        for (int i = 0; i < fQuickTransformed.size(); i++) {
            std::cout << (int) (fQuickTransformed[i]) << " ";
        }
        std::cout << ")" << std::endl;
        std::cout << "qtransform time => (" << qtime << "mcs )" << std::endl;
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