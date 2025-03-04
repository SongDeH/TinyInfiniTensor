#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 1. 去除冗余的transpose算子
        int TransposeCount = 0;
        auto slowPoint = ops.begin();
        std::ostringstream oss;
        auto fastPoint = ops.begin();
        while (fastPoint != ops.end())
        {
            if ((*fastPoint)->getOpType() == OpType::Transpose)
            {
                slowPoint = fastPoint;
                fastPoint++;
                if (fastPoint == ops.end())
                    break;
                if ((*fastPoint)->getOpType() == OpType::Transpose)
                {
                    if ((*slowPoint)->getOutputs()[0]->getGuid() == (*fastPoint)->getInputs()[0]->getGuid())
                    {
                        if (as<TransposeObj>(*fastPoint)->getPermute() == as<TransposeObj>(*slowPoint)->getPermute())
                        {
                            fastPoint = ops.erase(slowPoint); // erase returns iterator to next element
                            fastPoint = ops.erase(fastPoint); // erase and update fastPoint
                            fastPoint = ops.begin();          // reset iterator as requested in original code
                        }
                    }
                }
                else if ((*fastPoint)->getOpType() == OpType::MatMul)
                {
                    printf("matmul\n");
                    if ((*slowPoint)->getOpType() == OpType::Transpose)
                    {
                        auto permute = (as<TransposeObj>(*slowPoint))->getPermute();
                        int size = permute.size();
                        if (permute[size - 1] == (size - 2) && permute[size - 2] == size - 1)
                        {
                            if ((*slowPoint)->getOutputs()[0]->getGuid() == (*fastPoint)->getInputs()[0]->getGuid())
                            {
                                printf("transpose A\n");
                                as<MatmulObj>(*fastPoint)->setTransA(true);
                                // Start Generation Here
                                as<MatmulObj>(*fastPoint)->replaceInput((*fastPoint)->getInputs()[0], (*slowPoint)->getInputs()[0]);
                                ops.erase(slowPoint);
                            }
                            else if ((*slowPoint)->getOutputs()[0]->getGuid() == (*fastPoint)->getInputs()[1]->getGuid())
                            {
                                printf("transpose B\n");
                                as<MatmulObj>(*fastPoint)->setTransB(true);
                                // Start Generation Here
                                as<MatmulObj>(*fastPoint)->replaceInput((*fastPoint)->getInputs()[1], (*slowPoint)->getInputs()[0]);
                                ops.erase(slowPoint);
                            }
                        }
                    }
                }
                else
                {
                    fastPoint++;
                }
            }
        }
        std::unordered_set<int> usedFuid;
        for (const auto &op : ops)
        {
            for (const auto &input : op->getInputs())
                usedFuid.insert(input->getFuid());
            for (const auto &output : op->getOutputs())
                usedFuid.insert(output->getFuid());
        }
        // auto it = std::remove_if(tensors.begin(), tensors.end(),
        //                          [&](const Tensor &tensor)
        //                          { return usedFuid.find(tensor->getFuid()) == usedFuid.end(); });
        // tensors.erase(it, tensors.end());
        for(auto it = tensors.begin(); it != tensors.end(); )
        {
            if (usedFuid.find((*it)->getFuid()) == usedFuid.end()) // 如果tensor的fuid不在usedFuid中，则删除该tensor
                it = tensors.erase(it);
            else
                ++it;
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        // 为每个tensor分配内存
        std::unordered_map<int, size_t> tensorOffsets;  // 记录每个tensor的内存偏移量
        
        // 1. 为所有输入tensor分配内存
        for (auto &tensor : tensors) {
            if (!tensor->getSource()) {  // 输入tensor没有source
                size_t bytes = tensor->getBytes();
                size_t offset = allocator.alloc(bytes);
                tensorOffsets[tensor->getFuid()] = offset;
            }
        }

        // 2. 按拓扑顺序为算子的输出tensor分配内存
        for (auto &op : ops) {
            for (auto &output : op->getOutputs()) {
                size_t bytes = output->getBytes();
                size_t offset = allocator.alloc(bytes);
                tensorOffsets[output->getFuid()] = offset;
            }
        }

        // 3. 获取实际分配的内存指针并绑定到tensor
        void* basePtr = allocator.getPtr();
        for (auto &tensor : tensors) {
            auto fuid = tensor->getFuid();
            if (tensorOffsets.find(fuid) != tensorOffsets.end()) {
                size_t offset = tensorOffsets[fuid];
                char* tensorPtr = static_cast<char*>(basePtr) + offset;
                auto blob = make_ref<BlobObj>(runtime, tensorPtr);
                tensor->setDataBlob(blob);
            }
        }

        allocator.info();
        // =================================== 作业 ===================================
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini