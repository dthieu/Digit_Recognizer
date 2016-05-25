#include "Feature.h"
#include "Tools.h"

class CTesting{

	Models model;
public:
	CTesting(Models model){
		this->model = model;
	}

	/*Tạo kết quả thử nghiệm cho list Feature dựa vào dữ liệu sau khi train dataAfterTrain
	Kết quả trả về trong listResult*/
	void testListImage(string dataAfterTrain, Mat &listFeature, vector<int> &listResult);

	/*Tạo kết quả thử nghiệm cho 1 ảnh
	Kết quả trả về là lớp mà ảnh đó thuộc về sử dụng CvNormalBayesClassifier*/
	int testImage(CvNormalBayesClassifier &bayes, Mat& feature);//WithBayes

	/*Tạo kết quả thử nghiệm cho 1 ảnh
	Kết quả trả về là lớp mà ảnh đó thuộc về sử dụng CvNormalBayesClassifier*/
	int testImage(SVM &svm, Mat&);//WithSVM

	//Thay đổi mô hình thử nghiệm kết quả: SVM or Bayes
	void ChangeModels(Models model);

	
};