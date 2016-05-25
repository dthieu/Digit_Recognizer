#pragma once
#include <vector>
#include <fstream>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>

#include <opencv\ml.h>
using namespace std;
using namespace cv;

enum Rate{
	LOW,
	MEDIUM,
	HIGH
};

class CFeature
{
	int ratio;
public:

	/*	trích đặc trưng: Phương pháp HOG
		source: image
		feature: feature of source

	*/
	void HOGfeatureDetector( Mat &source, vector<float> &feature)
	{
		feature.clear();
		int i = 1;
		while ( (2 << i)  < source.rows )
			i++;

		int newRows = 2 << (i);

		Mat image;
		if ( newRows != source.rows)
			cv::resize(source, image,cv::Size(newRows,newRows));//resize ảnh về kích thước mới
		else
			image = source.clone();
		feature.clear();
		HOGDescriptor hog(Size(newRows,newRows), Size(newRows * 2/this->ratio,newRows * 2/this->ratio), Size(newRows/this->ratio,newRows/this->ratio), Size(newRows/this->ratio,newRows/this->ratio), 9);
		if (image.empty())
			throw new Exception();
		hog.compute(image, feature);//tạo vector đặc trưng feature từ Mat source
	
	}
	
	/*
	RateOfCellSize: the ratio between the WindowSize and CellSize
	CellSize = WindowSize / N.
	RateOfCellSize: LOW -> N = 4, MEDIUM -> N = 8, HIGH-> N = 16
	blockSize = CellSize * 2
	BlockStride = CellSize*/
	CFeature(Rate RateOfCellSize)
	{
		switch (RateOfCellSize)
		{
		case LOW:
			this->ratio = 2;
			break;
		case MEDIUM:
			this->ratio = 4;
			break;
		case HIGH:
			this->ratio = 8;
			break;
		default:
			this->ratio = 4;
			break;
		}

	}

	/*default: RateOfCellSize = LOW*/
	CFeature (){
		ratio = 4;
	}
	~CFeature()
	{

	}

	void HOGfeatureDetector( vector<Mat> &source, vector<vector<float>> &features){
		features.clear();

		for (unsigned int i = 0; i < source.size(); i++){
			vector<float> feature;
			this->HOGfeatureDetector(source[i], feature);
			features.push_back(feature);
		}
	}
};



