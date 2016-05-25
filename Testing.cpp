#include "Testing.h"

	void CTesting::testListImage(string dataAfterTrain, Mat& listFeatures, vector<int> &listResult){
		if (this->model == Models::_EMPTY)
			return;
		listResult.clear();
		listResult.resize(listFeatures.rows);
		CvNormalBayesClassifier bayes;
		SVM svm;

		if (this->model == Models::_BAYES){
			bayes.load(dataAfterTrain.c_str());

			for ( int i = 0; i < listFeatures.rows; i++){
				
				Mat tmp = cvCreateMat(1, listFeatures.cols, CV_32F);
				for ( int x = 0; x < tmp.cols; x++)
				{
					tmp.at<float>(0, x) = listFeatures.at<float>(i, x);
					float a = listFeatures.at<float>(i, x);
				}
				
				float result = bayes.predict(tmp, &Mat());
				listResult[i] = (int)result;
			}
			return;
		}

		if (this->model == Models::_SVM){

			svm.load(dataAfterTrain.c_str());
			
			for ( int i = 0; i < listFeatures.rows; i++){
				Mat tmp = cvCreateMat(1, listFeatures.cols, CV_32F);
				for ( int x = 0; x < tmp.cols; x++)
				{
					tmp.at<float>(0, x) = listFeatures.at<float>(i, x);
				}
				
				//int result = this->testImage(svm, tmp);
				float result = svm.predict(tmp);
				listResult[i] = (int)result;
			}
			return;
		}
	}

	int CTesting::testImage(CvNormalBayesClassifier &bayes, Mat& feature){//WithBayes
		float result = bayes.predict(feature, &Mat());
		return (int)(result);
	}

	int CTesting::testImage(SVM &svm, Mat& feature){//WithSVM
		float result = svm.predict(feature,false);
		return (int)(result);
	}

	void CTesting::ChangeModels(Models model){
		this->model = model;
	}