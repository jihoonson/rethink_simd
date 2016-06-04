#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <random>

#include <boost/program_options.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/singleton_pool.hpp>

namespace po = boost::program_options;

using namespace std;

const std::string FILE_NAME = "test.dat";

auto gen_random(char *s, const int len) -> void {
  static const char alphanum[] =
      "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len - 1; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  s[len - 1] = 0;
}

// schema: long, int, float, long, double, int, float, long, double, int, float, long, double, char(40), char[10], char(50), char[20] - 200 bytes
//            8   12     16    24      32   36     40    48      56   60     64    72      80       120       130       180       200

// schema: int, float - 8 bytes
//         key  payload

const size_t RECORD_SIZE = 8;
const size_t OUT_BUFFER_SIZE = 4096;

/*
 * Return a value of the key column
 */
auto get_key(uint8_t* buf) -> int32_t* {
  return reinterpret_cast<int32_t *>(buf); // + 8 for old schema
}

struct MMapFile {
  FILE* file;
  uint8_t* mm;
  size_t file_size;
};

struct PoolTag {};

typedef boost::singleton_pool<PoolTag, OUT_BUFFER_SIZE> pool;

auto read_file() -> std::shared_ptr<MMapFile> {
  std::shared_ptr<MMapFile> mm_file = std::make_shared<MMapFile>();
  mm_file->file = fopen(FILE_NAME.c_str(), "rb");
  fseek(mm_file->file, 0L, SEEK_END);
  mm_file->file_size = ftell(mm_file->file);
  void* result = mmap(nullptr, mm_file->file_size, PROT_READ, MAP_SHARED, fileno(mm_file->file), 0);
  mm_file->mm = reinterpret_cast<uint8_t*>(result);
  return mm_file;
}

struct OutBuffer {
  uint8_t* buf;
  size_t remain_size;

public:
  OutBuffer() {
    buf = reinterpret_cast<uint8_t*>(pool::malloc());
    remain_size = OUT_BUFFER_SIZE;
  }
};

auto write_to_out(std::shared_ptr<OutBuffer> out, uint8_t* in) -> std::shared_ptr<OutBuffer> {
  if (out->remain_size >= RECORD_SIZE) {
    memcpy(out->buf, in, RECORD_SIZE);
    out->remain_size -= RECORD_SIZE;
  } else {
    memcpy(out->buf, in, out->remain_size);
    size_t tail_size = RECORD_SIZE - out->remain_size;

    out->buf = reinterpret_cast<uint8_t*>(pool::malloc());
    memcpy(out->buf, in, tail_size);
    out->remain_size = OUT_BUFFER_SIZE - tail_size;
  }
  return out;
}

auto scalar_branch(size_t input_num, double_t lower_bound, double_t upper_bound) -> void {
  std::shared_ptr<MMapFile> in = read_file();

  std::shared_ptr<OutBuffer> out = std::make_shared<OutBuffer>();

  size_t j = 0;
  for (size_t i = 0; i < input_num; i++) {
    uint8_t *p = &in->mm[i * RECORD_SIZE];
    int32_t key = *get_key(p);

    if (key >= lower_bound && key < upper_bound) {
      out = write_to_out(out, p);
      j++;
    }
  }

  munmap(in->mm, in->file_size);
  fclose(in->file);

  pool::purge_memory();

  cout << "# of outputs: " << (j) << endl;
  cout << "selectivity: " << (j * 100 / (double_t)input_num) << "%\n";
}

auto scalar_branchless(size_t input_num, double_t lower_bound, double_t upper_bound) -> void {
  std::shared_ptr<MMapFile> in = read_file();

  std::shared_ptr<OutBuffer> out = std::make_shared<OutBuffer>();

  size_t j = 0;
  for (size_t i = 0; i < input_num; i++) {
    uint8_t *p = &in->mm[i * RECORD_SIZE];
    int32_t key = *get_key(p);

    out = write_to_out(out, p);
    int m = (key >= lower_bound) & (key < upper_bound);
    j += m;
  }

  munmap(in->mm, in->file_size);
  fclose(in->file);

  cout << "# of outputs: " << (j) << endl;
  cout << "selectivity: " << (j * 100 / (double_t)input_num) << "%\n";
}

auto generate_file(size_t input_num) -> void {
  srand(NULL);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> idis(0, input_num - 1);
  std::uniform_real_distribution<float> fdis(0, 1);
  std::uniform_real_distribution<> ddis(0, 1);

  // Random generation

  long l;
  int i;
  float f;
  double d;
  char ch[50];

  FILE* file = fopen(FILE_NAME.c_str(), "w");
  size_t file_size = RECORD_SIZE * input_num;
  cout << "file size: " << file_size << endl;
  if (ftruncate(fileno(file), file_size) == -1) {
    cerr << errno << endl;
  }
  fclose(file);

  file = fopen(FILE_NAME.c_str(), "r+b");
  fseek(file, 0L, SEEK_SET);

  void* result = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file), 0);
  if (result == MAP_FAILED) {
    cerr << errno << endl;
  }
  uint8_t* out = reinterpret_cast<uint8_t*>(result);

  for (size_t idx = 0; idx < input_num; idx++) {

    i = idis(gen);
    memcpy(out, &i, 4);
    out += 4;

    f = fdis(gen);
    memcpy(out, &f, 4);
    out += 4;

//    l = idis(gen);
//    memcpy(out, &l, 8);
//    out += 8;
//
//    for (int k = 0; k < 3; k++) {
//      i = idis(gen);
//      memcpy(out, &i, 4);
//      out += 4;
//
//      f = fdis(gen);
//      memcpy(out, &f, 4);
//      out += 4;
//
//      l = idis(gen);
//      memcpy(out, &l, 8);
//      out += 8;
//
//      d = ddis(gen);
//      memcpy(out, &d, 8);
//      out += 8;
//    }
//
//    gen_random(ch, 40);
//    memcpy(out, ch, 40);
//    out += 40;
//
//    gen_random(ch, 10);
//    memcpy(out, ch, 10);
//    out += 10;
//
//    gen_random(ch, 50);
//    memcpy(out, ch, 50);
//    out += 50;
//
//    gen_random(ch, 20);
//    memcpy(out, ch, 20);
//    out += 20;
  }

  cout << "written size: " << (out - reinterpret_cast<uint8_t*>(result)) << endl;

  munmap(result, file_size);
  fclose(file);
};

auto main(const int argc, char *argv[]) -> int32_t {
  size_t input_num = 5000000;
  double_t selectivity = 0.1f;

  po::options_description desc("rethink_simd\nusage");

  desc.add_options()
      ("help,h", "Display this help message.")

      ("generate,g",
       po::value<size_t>(&input_num)->value_name("INPUT_NUM"),
       "Generate data file.") //value_name("NUMBER")->default_value(1)

      ("selectivity,s",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY")->default_value(0.f),
       "Set selectivity")

      ("scalar_branch,b",
       po::value<size_t>(&input_num)->value_name("INPUT_NUM"),
       "Run scalar branch with a given selectivity")

      ("scalar_branchless,l",
       po::value<size_t>(&input_num)->value_name("INPUT_NUM"),
       "Run scalar branchless with a given selectivity");

  po::variables_map vm;
  auto optional_style = po::command_line_style::unix_style;

  po::store(po::parse_command_line(argc, argv, desc, optional_style), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 0;
  } else if (vm.count("generate")) {
    generate_file(input_num);
  } else if (vm.count("scalar_branch")) {
    double_t lower_bound = input_num / 2 - (input_num * selectivity / 2);
    double_t upper_bound = input_num / 2 + (input_num * selectivity / 2);
    printf("lower bound: %lf, upper bound: %lf\n", lower_bound, upper_bound);

    auto start = chrono::steady_clock::now();

    scalar_branch(input_num, lower_bound, upper_bound);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration <double, milli> (diff).count() << " ms" << endl;

  } else if (vm.count("scalar_branchless")) {
    double_t lower_bound = input_num / 2 - (input_num * selectivity / 2);
    double_t upper_bound = input_num / 2 + (input_num * selectivity / 2);
    printf("lower bound: %lf, upper bound: %lf\n", lower_bound, upper_bound);

    auto start = chrono::steady_clock::now();

    scalar_branchless(input_num, lower_bound, upper_bound);

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "Done! " << chrono::duration <double, milli> (diff).count() << " ms" << endl;

  } else {
    cout << "<Unknown option>\n" << desc << "\n";
  }

  return 0;
}