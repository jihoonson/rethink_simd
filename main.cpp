#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <random>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;

const std::string META_FILE = "test.meta";
const std::string KEY_FILE = "test.key";
const std::string PAYLOAD_FILE = "test.payload";

// old schema: long, int, float, long, double, int, float, long, double, int, float, long, double, char(40), char[10], char(50), char[20] - 200 bytes
//            8   12     16    24      32   36     40    48      56   60     64    72      80       120       130       180       200

// schema: int, float - 8 bytes
//         key  payload

const size_t OUT_BUFFER_SIZE = 4096;

struct MMapFile {
  FILE* file;
  uint8_t* mm;
  size_t file_size;

  auto release() -> void {
    munmap(mm, file_size);
    fclose(file);
  }
};

auto read_meta() -> size_t {
  size_t input_num;

  FILE* meta = fopen(META_FILE.c_str(), "rb");
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

  ScalarContext(double_t selectivity) {
    input_num = read_meta();

    cout << "input num: " << input_num << endl;

    lower_bound = input_num / 2 - (input_num * selectivity / 2);
    upper_bound = input_num / 2 + (input_num * selectivity / 2);

    std::shared_ptr<MMapFile> key_mm = read_file(KEY_FILE);
    std::shared_ptr<MMapFile> payload_mm = read_file(PAYLOAD_FILE);

    key_in = new int32_t[input_num];
    payload_in = new float_t[input_num];

    // Load whole data into memory
    memcpy(key_in, key_mm->mm, key_mm->file_size);
    memcpy(payload_in, payload_mm->mm, payload_mm->file_size);

    key_mm->release();
    payload_mm->release();

    key_out = new int32_t[selectivity * input_num > 0 ? (size_t)ceil(selectivity * input_num) : 1];
    payload_out = new float_t[selectivity * input_num > 0 ? (size_t)ceil(selectivity * input_num) : 1];
  }

  ~ScalarContext() {
    delete [] payload_in;
    delete [] key_in;
    delete [] payload_out;
    delete [] key_out;
  }

  size_t input_num;

  double_t lower_bound;
  double_t upper_bound;

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
      memcpy(&key_out[j], &key, 4);
      memcpy(&payload_out[j], &payload_in[i], 4);
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

    memcpy(&key_out[j], &key, 4);
    memcpy(&payload_out[j], &payload_in[i], 4);
    int m = (key >= lower_bound) & (key < upper_bound);
    j += m;
  }

  return j;
}

auto generate_file(size_t input_num) -> void {
  // Random generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> idis(0, input_num - 1);
  std::uniform_real_distribution<float> fdis(0, 1);

  int i;
  float f;

  FILE* meta_file = fopen(META_FILE.c_str(), "wb");
  fwrite(&input_num, sizeof(size_t), 1, meta_file);
  fclose(meta_file);

  FILE* key_file = fopen(KEY_FILE.c_str(), "w");
  FILE* payload_file = fopen(PAYLOAD_FILE.c_str(), "w");

  size_t file_size = input_num * 4; // for float and int
  cout << "file size: " << file_size << endl;
  if (ftruncate(fileno(key_file), file_size) == -1) {
    cerr << errno << endl;
  }
  if (ftruncate(fileno(payload_file), file_size) == -1) {
    cerr << errno << endl;
  }
  fclose(key_file);
  fclose(payload_file);

  key_file = fopen(KEY_FILE.c_str(), "r+b");
  payload_file = fopen(PAYLOAD_FILE.c_str(), "r+b");
  fseek(key_file, 0L, SEEK_SET);
  fseek(payload_file, 0L, SEEK_SET);

  void* key_mm = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(key_file), 0);
  if (key_mm == MAP_FAILED) {
    cerr << errno << endl;
  }
  uint8_t* key_out = reinterpret_cast<uint8_t*>(key_mm);

  void* payload_mm = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(payload_file), 0);
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

  munmap(key_mm, file_size);
  munmap(payload_mm, file_size);
  fclose(key_file);
  fclose(payload_file);
};

auto main(const int argc, char *argv[]) -> int32_t {
  size_t input_num;
  double_t selectivity;

  po::options_description desc("rethink_simd\nusage");

  desc.add_options()
      ("help,h", "Display this help message.")

      ("generate,g",
       po::value<size_t>(&input_num)->value_name("INPUT_NUM"),
       "Generate data file.")

      ("scalar_branch,b",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branch with a given selectivity")

      ("scalar_branchless,l",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branchless with a given selectivity");

  po::variables_map vm;
  auto optional_style = po::command_line_style::unix_style;

  po::store(po::parse_command_line(argc, argv, desc, optional_style), vm);
  po::notify(vm);

  if (vm.empty() || vm.count("help")) {
    cout << desc << "\n";

  } else if (vm.count("generate")) {
    generate_file(input_num);

  } else if (vm.count("scalar_branch")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(selectivity);

    auto start = chrono::steady_clock::now();

    size_t out_num = scalar_branch(context);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration <double, milli> (diff).count() << " ms" << endl;
    
    cout << "# of outputs: " << (out_num) << endl;
    cout << "selectivity: " << (out_num * 100 / (double_t)context->input_num) << "%\n";

  } else if (vm.count("scalar_branchless")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(selectivity);

    auto start = chrono::steady_clock::now();

    size_t out_num = scalar_branchless(context);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration <double, milli> (diff).count() << " ms" << endl;

    cout << "# of outputs: " << (out_num) << endl;
    cout << "selectivity: " << (out_num * 100 / (double_t)context->input_num) << "%\n";

  } else {
    cout << "<Unknown option>\n" << desc << "\n";

  }

  return 0;
}