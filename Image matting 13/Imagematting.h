#ifndef IMAGEMATTING_H
#define IMAGEMATTING_H

#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cmath>
#include <vector>
#include <opencv2/flann/matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
using namespace std;
using namespace cv;
using namespace Eigen;

#define K       10 // the nearst K neighbors(B/F) in RGBXY space
#define winSize 3 // the size of a neighbor matrix
#define sigma   0.1
#define delta   0.1
#define gamma   0.1
#define REG     1e-5

#define AT(mat, x, y) mat.at<int>(x, y)

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

struct kdResult // the result of knnsearch
{
	Mat      indices;
	Mat      dists;
};

class Imagematting
{
private:
	IplImage *img; // the original image
	IplImage *trimap; // the original trimap
	IplImage *matte; // the result image
	int      height;
	int      width;
	int      step; // widthstep of image
	int      g_step; // widthstep of gray
	int      channels;
	uchar*   data;
	int      N; // N = height * width + 2; order: VirtueF, VirtueB, all nodes ((0, 0), бнбн, (height - 1, width - 1))

	int      bsize; // size of background pixels
	int      fsize; // size of foreground pixels
	int      usize; // size of unknown pixels
	Mat      bmat; // mat of background pixels
	Mat      fmat; // mat of foreground pixels
	Mat      umat; // mat of unknown pixels
	Mat      allmat; // mat of all pixels
	int      *dB; // the min distance between C and B
	int      *dF; // the min distance between C and F
	int      **tri; // 1 is foreground, 0 is background, 2 is unknown
	double   **preAlpha; // mat of predicted alpha (n * 1 matrix)
        double   **confidence; // confidence for every pixel

	CvMat*   covarienceOfMat; // covarience Matrix in 3*3 window
	CvMat*   avgOfMat; // average Matrix in 3*3 window

	kdResult bresult, fresult, allresult; // the results of kd-tree by using FLANN

	// for eigen
	SpMat    W1; // save W(i, F) & W(i, B) in a big SparseMatrix W1(N * N)
	SpMat    W2; // save local smooth term Wlap(ij) in a big SparseMatrix W2(N * N)
	SpMat    W3; // save unlocal smooth term Wlle(ij) in a big SparseMatrix W3(N * N)
	SpMat    L;  // L = -W1 - W2 - W3
	SpMat    I;  // I is a diagonal Matrix
	VectorXd G;  // G is a Vector
	VectorXd Alpha; // the final alpha

	void     addInMat(Mat &mat, int n, int i, int j, int b, int g, int r); // add a RGBXY parameter in allmat
        void     addInMat(Mat &mat, int n, int x, int y); // add a XY parameter in  mat
	void     createMat(); // create b-, f-, u-, allmat

	void     getD(); //the minimum distances between foreground/background sample and the current pixel
	double   geteveryAlpha(int c, int f, int b); // get predicted alpha of C, use it fth-nearest foreground pixel and bth-nearest background pixel
	double   getRd(int c, int f, int b); // get Rd of C, use it fth-nearest foreground pixel and bth-nearest background pixel
	double   getW(int c, int fb, bool flag); // get W of C, fb is the fbth-nearest fore- or background pixel of C
	double   getConfidence(int c, int f, int b);  // c is the index in umat, f is the index of C's f-nearest foreground pixel,  b is the index of C's b-nearest background pixel

	void     getWeight1(); // get data term W(i, F) & W(i, B)
	void     getWeight2(); // get local smooth term Wlap(ij)
	void     getWeight3(); // get unlocal smooth term Wlle(ij)

	void     getCovarianceMatrix(int i, int j); // get covariance matrix of a 3*3 window, x & y are the middle points in one 3*3 window
	void     getCiCj(CvMat *mat, int i, int j);
	void     getG();
	void     getI();
	void     getL(); // get L

        int      BC(Mat &mat, int index);
        int      GC(Mat &mat, int index);
        int      RC(Mat &mat, int index);

public:
	Imagematting();
	~Imagematting();

	void     loadImage(char * filename);
	void     loadTrimap(char * filename);

	void     findKnearest(); // find K nearest pixels of every pixel
	void     getPreAlpha(); // get predicted Alpha Matrix named preAlpha
	void     getFinalAlpha();  // use QR to solve Ax = b

	void     save(char * filename);
	void     solveAlpha();
	void     showMatte();

        //test

        void TEST(SpMat A);
        void testgetWeight2();
};

#endif //IMAGEMATTING_H
