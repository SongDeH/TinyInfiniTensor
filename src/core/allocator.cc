#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // 如果free_blocks为空，直接在末尾分配
        if (free_blocks.empty())
        {
            size_t addr_offset = used;
            used += size;
            peak = std::max(peak, used);
            return addr_offset;
        }

        // 寻找最佳适配的空闲块
        auto best_fit = free_blocks.end();
        size_t min_waste = SIZE_MAX;

        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it)
        {
            if (it->second >= size)
            {
                size_t waste = it->second - size;//计算浪费的空间
                if (waste < min_waste)//如果浪费的空间小于min_waste，则更新best_fit
                {
                    min_waste = waste;
                    best_fit = it;
                }
            }
        }

        // 如果找到合适的块
        if (best_fit != free_blocks.end())
        {
            size_t addr_offset = best_fit->first;
            size_t block_size = best_fit->second;

            // 从free_blocks中移除这个块
            free_blocks.erase(best_fit);

            // 如果剩余空间足够大，添加新的空闲块
            if (block_size > size)
            {
                free_blocks[addr_offset + size] = block_size - size;
            }

            return addr_offset;
        }

        // 如果没有找到合适的块，在末尾分配
        size_t addr_offset = used;
        used += size;
        peak = std::max(peak, used);
        return addr_offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        
        // 先将当前块加入free_blocks
        free_blocks[addr] = size;

        // 尝试向后合并
        auto next = free_blocks.find(addr + size);
        if (next != free_blocks.end())
        {
            // 合并当前块和后一个块
            free_blocks[addr] += next->second;
            free_blocks.erase(next);
        }

        // 尝试向前合并
        auto it = free_blocks.begin();
        while (it != free_blocks.end())
        {
            if (it->first + it->second == addr)
            {
                // 合并前一个块和当前块
                size_t merged_size = it->second + free_blocks[addr];
                free_blocks[it->first] = merged_size;
                free_blocks.erase(addr);
                break;
            }
            ++it;
        }

        // 如果这个块在末尾，可以减少used的大小
        auto last_block = free_blocks.rbegin();
        if (last_block != free_blocks.rend() &&
            last_block->first + last_block->second == used)
        {
            used = last_block->first;
            free_blocks.erase(last_block->first);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
