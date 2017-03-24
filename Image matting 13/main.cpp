#include "Imagematting.h"
#include <time.h>
#include <string>

int main(int argc, char **argv)
{
  Imagematting sm;
  if (argc != 4) cout << "Usage: ./main input_image_in_pic input_trimap_in_pic output_image_in_results\n";
  else
    {
      clock_t start, finish;
      start = clock();
      string input_image = string("pic/") + argv[1];
      string input_trimap = string("pic/") + argv[2];
      string output_image = string("results/") + argv[3];
      sm.loadImage(const_cast<char*>(input_image.c_str()));
      sm.loadTrimap(const_cast<char*>(input_trimap.c_str()));
      sm.solveAlpha();
      sm.save(const_cast<char*>(output_image.c_str()));
      finish = clock();
      cout << double(finish - start) / CLOCKS_PER_SEC << endl;
    }
  return 0;
}

