#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto dim_a = inputs[0]->getDims().size();
        auto dim_b = inputs[1]->getDims().size();
        IT_ASSERT(dim_a == dim_b, "Matmul: input dimensions mismatch");
        auto output_shape = inputs[0]->getDims();
        auto output_dim = output_shape.size();
        int k2;
        if (transA)
        {
            m = inputs[0]->getDims()[output_dim - 1];
            k = inputs[0]->getDims()[output_dim - 2];
        }
        else
        {
            m = inputs[0]->getDims()[output_dim - 2];
            k = inputs[0]->getDims()[output_dim - 1];
        }
        if (transB)
        {
            n = inputs[1]->getDims()[output_dim - 2];
            k2 = inputs[1]->getDims()[output_dim - 1];
        }
        else
        {
            n = inputs[1]->getDims()[output_dim - 1];
            k2 = inputs[1]->getDims()[output_dim - 2];
        }
        IT_ASSERT(k == k2, "Matmul: input dimensions mismatch");
        for (size_t i = 0; i < dim_a - 2; i++)
        {
            output_shape[i] = std::max(inputs[0]->getDims()[i], inputs[1]->getDims()[i]);
        }
        output_shape[output_dim - 1] = n;
        output_shape[output_dim - 2] = m;

        return {{output_shape}};
    }

} // namespace infini