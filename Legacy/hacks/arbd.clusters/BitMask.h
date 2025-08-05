#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <initializer_list>


class BitMaskBase {
public:
    BitMaskBase(const size_t len) : len(len) {}
    __host__ __device__
    virtual void set_mask(size_t i, bool value) = 0;
    __host__ __device__
    virtual bool get_mask(size_t i) const = 0;
    size_t get_len() const { return len; }
    virtual void print() {
	for (int i = 0; i < len; ++i) {
	    printf("%d", (int) get_mask(i));
	}
	printf("\n");
    }
    
protected:
    size_t len;
};

// Don't use base because virtual member functions require device malloc
class BitMask {
    typedef size_t idx_t;
    typedef unsigned int data_t;
public:
    BitMask(const idx_t len) : len(len) {
	idx_t tmp = get_array_size();
	// printf("len %lld\n",len);
	// printf("tmp %d\n",tmp);
	assert(tmp * data_stride >= len);
	mask = (tmp > 0) ? new data_t[tmp] : NULL;
	for (int i = 0; i < tmp; ++i) mask[i] = data_t(0);
    }
    ~BitMask() { if (mask != NULL) delete[] mask; }

    __host__ __device__
    idx_t get_len() const { return len; }

    __host__ __device__
    void set_mask(idx_t i, bool value) {
	// return;
	assert(i < len);
	idx_t ci = i/data_stride;
	data_t change_bit = (data_t(1) << (i-ci*data_stride));
#ifdef __CUDA_ARCH__
	if (value) {
	    atomicOr(  &mask[ci], change_bit );
	} else {
	    atomicAnd( &mask[ci], ~change_bit );
	}
#else
	if (value) {
	    mask[ci] = mask[ci] | change_bit;
	} else {
	    mask[ci] = mask[ci] & (~change_bit);
	}
#endif
    }

    __host__ __device__
    bool get_mask(const idx_t i) const {
	// return false;
	assert(i < len);
	const idx_t ci = i/data_stride;
	return mask[ci] & (data_t(1) << (i-ci*data_stride));
    }

    __host__
    BitMask* copy_to_cuda() const {
	BitMask* obj_d = NULL;
	BitMask obj_tmp(0);
	data_t* mask_d = NULL;
	size_t sz = sizeof(data_t) * get_array_size();
	gpuErrchk(cudaMalloc(&obj_d, sizeof(BitMask)));
	if (sz > 0) {
	    gpuErrchk(cudaMalloc(&mask_d, sz));
	    gpuErrchk(cudaMemcpy(mask_d, mask, sz, cudaMemcpyHostToDevice));
	}
	// printf("BitMask::copy_to_cuda() len(%lld) mask(%x)\n", len, mask_d);
	obj_tmp.len = len;
	obj_tmp.mask = mask_d;
	gpuErrchk(cudaMemcpy(obj_d, &obj_tmp, sizeof(BitMask), cudaMemcpyHostToDevice));
	obj_tmp.mask = NULL;
	return obj_d;
    }

    __host__
    static void remove_from_cuda(BitMask* obj_d) {
	BitMask obj_tmp(0);
	gpuErrchk(cudaMemcpy(&obj_tmp, obj_d, sizeof(BitMask), cudaMemcpyDeviceToHost));
	if (obj_tmp.len > 0) {
	    gpuErrchk(cudaFree(obj_tmp.mask));
	}
	obj_tmp.mask = NULL;
	gpuErrchk(cudaMemset((void*) &(obj_d->mask), (int) NULL, sizeof(data_t*))); // set NULL on to device
	gpuErrchk(cudaFree(obj_d));
	obj_d = NULL;
    }

    __host__ __device__
    void print() {
	for (int i = 0; i < len; ++i) {
	    printf("%d", (int) get_mask(i));
	}
	printf("\n");
    }

private:
    idx_t get_array_size() const { return (len == 0) ? 1 : (len-1)/data_stride + 1; }
    idx_t len;
    const static idx_t data_stride = CHAR_BIT * sizeof(data_t)/sizeof(char);
    data_t* __restrict__ mask;
};

/*
template<size_t chunk_size>
class SparseBitMask : public BitMaskBase {
public:
    SparseBitMask(const size_t len) : BitMaskBase(len) {
	assert( sparse_size > 0 );
	meta_len = (len-1)/data_stride + 1;
	meta_mask = BitMask(meta_len);
    }
    ~BitMask() { delete[] mask; }

    void set_mask(size_t i, bool value) {
	assert(i < len);
	size_t meta_i = i/data_stride;
	// TODO: keep track of
	
	// bool is_set = get_mask(i);
	// if ((!is_set) && value) {
	//     mask[ci] = mask[ci] + (1 << (i-ci*data_stride));
	// } else if (is_set && (!value)) {
	//     mask[ci] = mask[ci] - (1 << (i-ci*data_stride));
	// }
    }
    size_t meta_idx_to_char_idx(size_t meta_idx) {
	return meta_idx * CHAR_BIT;
    }

    bool get_mask(size_t i) {
	// assert(i < len);
	// size_t ci = i/data_stride;
	// return mask[ci] & (1 << (i-ci*data_stride));
    }
    
private:
    const static size_t data_stride = CHAR_BIT*spare_size;
    size_t meta_len;
    BitMask meta_mask;
    std::vector<char> mask;
};
*/

bool test_BitMask() {
    for (int i: {8,9,20}) {
	BitMask b = BitMask(i);
	for (int j: {0,3,10,19}) {
	    if (j < i) b.set_mask(j,1);
	}
	printf("Testing BitMask(%d)\n", i);
	b.print();
	b.set_mask(3,1);
	b.print();
	b.set_mask(3,1);
	b.print();
	b.set_mask(3,0);
	b.print();
	b.set_mask(2,0);
	b.print();
    }
    return true;
}
// */

/*
int main(int argc, char* argv[]) {
    test_BitMask();
    return 1;
}
//*/
