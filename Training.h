#pragma once
#include "Feature.h"
#include "Tools.h"

class CTrainingData{
	Models model;
	string fileResultAfterTrain;
public:
	CTrainingData(string fileResultAfterTrain);
	~CTrainingData();

	
	bool  TrainingForBayes(Mat &features, Mat&Labels);
	bool  TrainingForSVM(Mat &features, Mat& labels, CvSVMParams param);

	bool emptyTrain();

	Models getCurrentModels();
};