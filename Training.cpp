#include "Training.h"



CTrainingData::CTrainingData(string fileResultAfterTrain){
	if (fileResultAfterTrain.length() == 0)
	{
		fileResultAfterTrain = "";
		this->model = Models::_EMPTY;
		
	}

	this->fileResultAfterTrain = fileResultAfterTrain;
}

CTrainingData::~CTrainingData(){
}

bool CTrainingData::TrainingForBayes(Mat &features, Mat& labels){
	this->model = Models::_BAYES;
	CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier();
	bool result = bayes->train(features, labels, Mat(), Mat(), false);
	if (!result)
		return false;

	bayes->save(fileResultAfterTrain.c_str());
	delete bayes;
	return true;

}

bool CTrainingData::TrainingForSVM(Mat &features, Mat& labels, CvSVMParams param){
	this->model = Models::_SVM;
	
	SVM *svm = new SVM();
	bool result = svm->train(features, labels, Mat(), Mat(), param);
	if (!result)
		return false;

	svm->save(this->fileResultAfterTrain.c_str());
	delete svm;
	return true;


}

bool CTrainingData::emptyTrain(){
	if(this->model == Models::_EMPTY)
		return true;
	return false;
}

Models CTrainingData::getCurrentModels(){
	return this->model;
}