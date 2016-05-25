#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv\cv.h>

using namespace std;
using namespace cv;

static int N_ROWS = 28;
static int N_COLS = 28;


enum Models{
	_BAYES,
	_SVM,
	_EMPTY
};

class CTools{

public:
	/*Hàm đọc dữ liệu ảnh từ File fileName
	Ảnh được lưu vào dataTrain
	Nếu có nhãn thì sẽ được lưu trong labels */
	static bool ReadDataTrain (string fileNameIn, vector<Mat> &dataTrain, vector<int>& labels);

	/*Hàm đọc dữ liệu ảnh từ File fileName
	Ảnh được lưu vào dataTest
	Lưu ý: DataTest nên dữ liệu không có nhãn*/
	static bool ReadDataTest  (string fileNameIn, vector<Mat> &dataTest);


	/*Hàm ghi dữ liệu từ result ra fileNameOut theo định dạng
	"ImageId", "Label"
	1		,  "x"
	2		,  "y"
	...		,  "..."
	*/
	static void WriteResult(vector<int> result, string fileNameOut);

	static bool read_Mnist(string filename, vector<Mat> &vec);

	static bool read_Mnist_Label(string filename, vector<int> &vec);

	static void ConvertListFeatureToMat(vector<vector<float>> &, Mat &);

	static void Merge(vector<vector<float>> &, vector<vector<float>> &, Mat&);

	static Mat CombinedImage (vector<Mat> listImage,unsigned int number);
};


