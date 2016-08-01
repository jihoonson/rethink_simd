#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <random>
#include <climits>

#include <boost/program_options.hpp>

#include "immintrin.h"
#include "emmintrin.h"

namespace po = boost::program_options;

using namespace std;

const size_t PAYLOAD_SIZE = 4;

const std::string META_FILE = "test.meta";
const std::string KEY_FILE = "test.key";
const std::string PAYLOAD_FILE = "test.payload";

// old schema: long, int, float, long, double, int, float, long, double, int, float, long, double, char(40), char[10], char(50), char[20] - 200 bytes
//            8   12     16    24      32   36     40    48      56   60     64    72      80       120       130       180       200

// schema: int, float - 8 bytes
//         key  payload

struct MMapFile {
  FILE* file;
  uint8_t* mm;
  size_t file_size;

  auto release() -> void {
    munmap(mm, file_size);
    fclose(file);
  }
};

auto read_meta(std::string data_dir) -> size_t {
  size_t input_num;

  FILE* meta = fopen((data_dir + META_FILE).c_str(), "rb");
  fread(&input_num, sizeof(size_t), 1, meta);
  fclose(meta);

  return input_num;
}

auto read_file(std::string file_name) -> std::shared_ptr<MMapFile> {
  std::shared_ptr<MMapFile> mm_file = std::make_shared<MMapFile>();

  mm_file->file = fopen(file_name.c_str(), "rb");
  fseek(mm_file->file, 0L, SEEK_END);
  mm_file->file_size = ftell(mm_file->file);

  void* result = mmap(nullptr, mm_file->file_size, PROT_READ, MAP_SHARED, fileno(mm_file->file), 0);
  if (result == MAP_FAILED) {
    return nullptr;
  }

  mm_file->mm = reinterpret_cast<uint8_t*>(result);

  return mm_file;
}

struct ScalarContext {

  ScalarContext(std::string data_dir, double_t selectivity) {
    input_num = read_meta(data_dir);

    cout << "input num: " << input_num << endl;

    lower_bound = input_num / 2 - (input_num * selectivity / 2);
    upper_bound = input_num / 2 + (input_num * selectivity / 2);

    std::shared_ptr<MMapFile> key_mm = read_file(data_dir + KEY_FILE);
    std::shared_ptr<MMapFile> payload_mm = read_file(data_dir + PAYLOAD_FILE);

    key_in = new int32_t[input_num];
    payload_in = new float_t[input_num];

    // Load whole data into memory
    memcpy(key_in, key_mm->mm, key_mm->file_size);
    memcpy(payload_in, payload_mm->mm, payload_mm->file_size);

    key_mm->release();
    payload_mm->release();

    key_out = new int32_t[input_num];
    payload_out = new float_t[input_num];
  }

  ~ScalarContext() {
    delete [] payload_in;
    delete [] key_in;
    delete [] payload_out;
    delete [] key_out;
  }

  size_t input_num;

  float_t lower_bound;
  float_t upper_bound;

  int32_t* key_in;
  float_t* payload_in;

  int32_t* key_out;
  float_t* payload_out;
};

auto scalar_branch(std::shared_ptr<ScalarContext> context) -> size_t {
  size_t input_num = context->input_num;
  int32_t* key_in = context->key_in;
  float_t* payload_in = context->payload_in;
  double_t lower_bound = context->lower_bound;
  double_t upper_bound = context->upper_bound;
  int32_t* key_out = context->key_out;
  float_t* payload_out = context->payload_out;

  size_t j = 0;

  for (size_t i = 0; i < input_num; i++) {
    int32_t key = key_in[i];

    if (key >= lower_bound && key < upper_bound) {
      key_out[j] = key;
      payload_out[j] = payload_in[i];
      j++;
    }
  }

  return j;
}

auto scalar_branchless(std::shared_ptr<ScalarContext> context) -> size_t {
  size_t input_num = context->input_num;
  int32_t* key_in = context->key_in;
  float_t* payload_in = context->payload_in;
  double_t lower_bound = context->lower_bound;
  double_t upper_bound = context->upper_bound;
  int32_t* key_out = context->key_out;
  float_t* payload_out = context->payload_out;

  size_t j = 0;

  for (size_t i = 0; i < input_num; i++) {
    int32_t key = key_in[i];

    key_out[j] = key;
    payload_out[j] = payload_in[i];
    int m = (key >= lower_bound ? 1 : 0) & (key < upper_bound ? 1 : 0);
    j += m;
  }

  return j;
}

static void printvs(const char * label, __m256 v)
{
  float_t a[8];
  _mm256_store_ps(a, v);
  printf("%s = %f %f %f %f %f %f %f %f\n", label, a[0],  a[1],  a[2],  a[3], a[4],  a[5],  a[6],  a[7]);
}

static const __m256i _m256_mask = _mm256_set1_epi32(0xffffffff);
static const __m128i _m128_mask = _mm_set1_epi32(0xffffffff);

static void printvi256(const char * label, __m256i v)
{
  int32_t a[8];
  _mm256_maskstore_epi32(a, _m256_mask, v);
  printf("%s = %d %d %d %d %d %d %d %d\n", label, a[0],  a[1],  a[2],  a[3], a[4],  a[5],  a[6],  a[7]);
}

static void printvi128(const char * label, __m128i v)
{
  int32_t a[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(a), v);
  printf("%s = %d %d %d %d\n", label, a[0],  a[1],  a[2],  a[3]);
}

static const size_t RID_BUF_SIZE = 4 * 1024 * 1024 / 4;

auto prepare_perm_mat(__m128i* perm) -> __m128i* {
  for (uint16_t i = 0; i < 256; i++) {
    std::bitset<8> bits(i);
    std::vector<uint16_t> on;
    std::vector<uint16_t> off;

    for (uint16_t j = 0; j < 8; j++) {
      if (bits[j]) {
        on.push_back(j);
      } else {
        off.push_back(j);
      }
    }

    uint16_t a[8];
    int j = 0;
    for (std::vector<uint16_t>::iterator it = on.begin(); it != on.end(); ++it, ++j) {
      a[j] = *it;
    }
    for (std::vector<uint16_t>::iterator it = off.begin(); it != off.end(); ++it, ++j) {
      a[j] = *it;
    }
    perm[i] = _mm_set_epi16(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
  }
  return perm;
}

auto perform_vector(__m256i rid, __m128i *perm_mat, __m256i key, __m256 lb, __m256 ub,
                    int32_t* rid_buf, size_t& buf_idx, size_t& buf_start_idx,
                    int32_t* key_in, float_t* payload_in, int32_t* key_out, float_t* payload_out) -> void {

  __m256 cvt_key = _mm256_cvtepi32_ps(key);

  // Unordered compare checks that either inputs are NaN or not. Ordered compare checks that neither inputs are NaN.
  __m256 lb_cmp = _mm256_cmp_ps(lb, cvt_key, _CMP_NGT_US); // If lb is less than key, result is 0xFFFFFFFF. Otherwise, 0.
  __m256 ub_cmp = _mm256_cmp_ps(cvt_key, ub, _CMP_NGE_US);
  __m256 cmp = _mm256_and_ps(lb_cmp, ub_cmp); // if any element is 0xFFFFFFFF, then key satisfies the predicates.

  __mmask8 mask = (__mmask8) _mm256_movemask_ps(cmp);

  if (mask > 0 /* if any bit is set */) {
    // selective store
//      __m128i perm_comp = _mm_loadl_epi64(&perm_mat[mask]);
    __m128i perm_comp = perm_mat[mask];
    __m256i perm = _mm256_cvtepi16_epi32(perm_comp);

    // permute and store the input pointers
    __m256i cvt_cmp = _mm256_cvtps_epi32(cmp);
    cvt_cmp = _mm256_permutevar8x32_epi32(cvt_cmp, perm);
    __m256i ptr = _mm256_permutevar8x32_epi32(rid, perm);

    _mm256_maskstore_epi32(&rid_buf[buf_idx], cvt_cmp, ptr);

    buf_idx += _mm_popcnt_u64(mask);

    // if the buffer is full, flush the buffer
    if (buf_idx + 8 > RID_BUF_SIZE) {
      size_t b;
      for (b = 0; b + 8 < buf_idx; b += 8) {
        // dereference column values and store
        __m256i load_ptr = _mm256_load_si256(reinterpret_cast<__m256i*>(&rid_buf[b]));
        __m256i gather_key = _mm256_i32gather_epi32(key_in, load_ptr, 4);
        __m256 gather_pay = _mm256_i32gather_ps(payload_in, load_ptr, 4);

        // streaming store
        _mm256_stream_si256(reinterpret_cast<__m256i*>(&key_out[b + buf_start_idx]), gather_key);
        _mm256_stream_ps(&payload_out[b + buf_start_idx], gather_pay);
      }

      // Move extra items to the start of the buffer
      ptr = _mm256_load_si256(reinterpret_cast<__m256i*>(&rid_buf[b]));
      _mm256_store_si256(reinterpret_cast<__m256i*>(&rid_buf[0]), ptr);

      buf_start_idx += b;
      buf_idx -= b;
    }
  }

}

auto run_vector(std::shared_ptr<ScalarContext> context) -> size_t {
  // extracting one bit at a time from the bitmask, or use vector selective stores
  // early vs late materialization

  size_t input_num = context->input_num;
  int32_t* key_in = context->key_in;
  float_t* payload_in = context->payload_in;
  float_t lower_bound = context->lower_bound;
  float_t upper_bound = context->upper_bound;

  size_t key_out_sz = sizeof(int32_t) * input_num;
  void* key_out_buf = context->key_out;
  int32_t* key_out = reinterpret_cast<int32_t *>(std::align(32, // for streaming store
                                                            4,  // size of int32_t
                                                            key_out_buf,
                                                            key_out_sz));

  size_t pay_out_sz = sizeof(float_t) * input_num;
  void* pay_out_buf = context->payload_out;
  float_t* payload_out = reinterpret_cast<float_t*>(std::align(32, // for streaming store
                                                               4,  // size of float_t
                                                               pay_out_buf,
                                                               pay_out_sz));

  __m256 lb = _mm256_broadcast_ss(&lower_bound);
  __m256 ub = _mm256_broadcast_ss(&upper_bound);

  int32_t rid_buf[RID_BUF_SIZE]; // TODO: should be cache resident
  size_t buf_idx = 0;
  size_t buf_start_idx = 0;

  __m256i v_rid = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  const __m256i v_eight = _mm256_set_epi32(8, 8, 8, 8, 8, 8, 8, 8);

  const size_t remain_iter = input_num % 8;
  const size_t max_iter = input_num - remain_iter;

  __m128i perm_mat[256];
  prepare_perm_mat(perm_mat);

//  cout << "buf_start_idx: " << buf_start_idx << " buf_idx: " << buf_idx << endl;
  size_t rid;
  for (rid = 0; rid < max_iter; rid += 8) {
    __m256i v_key = _mm256_load_si256(reinterpret_cast<__m256i*>(&key_in[rid]));
    perform_vector(v_rid, perm_mat, v_key, lb, ub, rid_buf, buf_idx, buf_start_idx, key_in, payload_in, key_out, payload_out);

    v_rid = _mm256_add_epi32(v_rid, v_eight);
  }

  // evaluate remaining keys
  int32_t remain_key_arr[8] = {-1};
  for (int i = 0; i < remain_iter; i++) {
    remain_key_arr[i] = key_in[rid + i];
  }
  __m256i remain_key = _mm256_set_epi32(remain_key_arr[7], remain_key_arr[6], remain_key_arr[5], remain_key_arr[4],
                                        remain_key_arr[3], remain_key_arr[2], remain_key_arr[1], remain_key_arr[0]);
  // TODO: mask?
  perform_vector(v_rid, perm_mat, remain_key, lb, ub, rid_buf, buf_idx, buf_start_idx, key_in, payload_in, key_out, payload_out);

  // flush remaining items in the buffer
  size_t b = 0;
  for (b = 0; b + 8 < buf_idx; b += 8) {
    // dereference column values and store
    __m256i ptr = _mm256_load_si256(reinterpret_cast<__m256i*>(&rid_buf[b]));
    __m256i key = _mm256_i32gather_epi32(key_in, ptr, 4);
    __m256 pay = _mm256_i32gather_ps(payload_in, ptr, 4);

    // streaming store
    _mm256_stream_si256(reinterpret_cast<__m256i*>(&key_out[b + buf_start_idx]), key);
    _mm256_stream_ps(&payload_out[b + buf_start_idx], pay);
  }

  // flush remaining items in the buffer
  uint32_t v_mask[8] = {0};
  for (int i = 0; i < buf_idx - b; i++) {
    v_mask[i] = 0xFFFFFFFF;
  }
  __m256i remain_mask = _mm256_set_epi32(v_mask[7], v_mask[6], v_mask[5], v_mask[4], v_mask[3], v_mask[2], v_mask[1], v_mask[0]);
  const __m256i zero = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);

  __m256i ptr = _mm256_maskload_epi32(&rid_buf[b], remain_mask);
  __m256i key = _mm256_mask_i32gather_epi32(zero, key_in, ptr, remain_mask, 4);
  __m256 pay = _mm256_mask_i32gather_ps(zero, payload_in, ptr, remain_mask, 4);

  _mm256_maskstore_epi32(&key_out[b + buf_start_idx], remain_mask, key);
  _mm256_maskstore_ps(&payload_out[b + buf_start_idx], remain_mask, pay);

  return buf_start_idx + buf_idx;
}

auto generate_file(std::string data_dir, size_t input_num) -> void {
  // Random generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> idis(0, input_num - 1);
  std::uniform_real_distribution<float> fdis(0, 1);

  int i;
  float f;

  FILE* meta_file = fopen((data_dir + META_FILE).c_str(), "wb");
  fwrite(&input_num, sizeof(size_t), 1, meta_file);
  fclose(meta_file);

  FILE* key_file = fopen((data_dir + KEY_FILE).c_str(), "w");
  FILE* payload_file = fopen((data_dir + PAYLOAD_FILE).c_str(), "w");

  size_t key_file_size = input_num * 4; // for int
  cout << "key file size: " << key_file_size << endl;
  if (ftruncate(fileno(key_file), key_file_size) == -1) {
    cerr << errno << endl;
  }
  size_t payload_file_size = input_num * PAYLOAD_SIZE;
  cout << "payload file size: " << payload_file_size << endl;
  if (ftruncate(fileno(payload_file), payload_file_size) == -1) {
    cerr << errno << endl;
  }
  fclose(key_file);
  fclose(payload_file);

  key_file = fopen((data_dir + KEY_FILE).c_str(), "r+b");
  payload_file = fopen((data_dir + PAYLOAD_FILE).c_str(), "r+b");
  fseek(key_file, 0L, SEEK_SET);
  fseek(payload_file, 0L, SEEK_SET);

  void* key_mm = mmap(nullptr, key_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(key_file), 0);
  if (key_mm == MAP_FAILED) {
    cerr << errno << endl;
  }
  uint8_t* key_out = reinterpret_cast<uint8_t*>(key_mm);

  void* payload_mm = mmap(nullptr, payload_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(payload_file), 0);
  if (payload_mm == MAP_FAILED) {
    cerr << errno << endl;
  }
  uint8_t* payload_out = reinterpret_cast<uint8_t *>(payload_mm);

  // write date in columnar format
  for (size_t idx = 0; idx < input_num; idx++) {
    i = idis(gen);
    memcpy(key_out, &i, 4);
    key_out += 4;

    f = fdis(gen);
    memcpy(payload_out, &f, 4);
    payload_out += 4;
  }

  cout << "written key: " << (key_out - reinterpret_cast<uint8_t*>(key_mm)) << endl;
  cout << "written payload: " << (payload_out - reinterpret_cast<uint8_t*>(payload_mm)) << endl;

  munmap(key_mm, key_file_size);
  munmap(payload_mm, payload_file_size);
  fclose(key_file);
  fclose(payload_file);
};

auto main(const int argc, char *argv[]) -> int32_t {
  size_t input_num;
  double_t selectivity;
  std::string directory;

  po::options_description desc("rethink_simd\nusage");

  desc.add_options()
      ("help,h", "Display this help message.")

      ("path,p",
       po::value<std::string>(&directory)->value_name("DATA_PATH")->default_value("./"),
       "Directory path to data file.")

      ("generate,g",
       po::value<size_t>(&input_num)->value_name("INPUT_NUM"),
       "Generate data file.")

      ("scalar_branch,b",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branch with a given selectivity")

      ("scalar_branchless,l",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branchless with a given selectivity")

      ("vector,v",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run vector with a given selectivity");

  po::variables_map vm;
  auto optional_style = po::command_line_style::unix_style;

  po::store(po::parse_command_line(argc, argv, desc, optional_style), vm);
  po::notify(vm);

  if (vm.count("generate")) {
    generate_file(directory, input_num);

  } else if (vm.count("scalar_branch")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = chrono::steady_clock::now();

    size_t out_num = scalar_branch(context);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration <double, milli> (diff).count() << " ms" << endl;

    cout << "# of outputs: " << (out_num) << endl;
    cout << "selectivity: " << (out_num * 100 / (double_t)context->input_num) << "%\n";

  } else if (vm.count("scalar_branchless")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = chrono::steady_clock::now();

    size_t out_num = scalar_branchless(context);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration<double, milli>(diff).count() << " ms" << endl;

    cout << "# of outputs: " << (out_num) << endl;
    cout << "selectivity: " << (out_num * 100 / (double_t) context->input_num) << "%\n";

  } else if (vm.count("vector")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = chrono::steady_clock::now();

    size_t out_num = run_vector(context);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration<double, milli>(diff).count() << " ms" << endl;

    cout << "# of outputs: " << (out_num) << endl;
    cout << "selectivity: " << (out_num * 100 / (double_t) context->input_num) << "%\n";
  } else {
    cout << desc << "\n";

  }

  return 0;
}