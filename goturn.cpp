#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/datasets/track_alov.hpp>
#include <opencv2/datasets/track_vot.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>


using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;



#define INPUT_SIZE 227
#define NUM_CHANNELS 3

Rect2f points2rect(vector<Point2d> gtPoints)
{
	float minX = 99999, maxX = 0, minY = 99999, maxY = 0;
	for (int j = 0; j < (int)gtPoints.size(); j++)
	{
		if (maxX < gtPoints[j].x) maxX = gtPoints[j].x;
		if (maxY < gtPoints[j].y) maxY = gtPoints[j].y;
		if (minX > gtPoints[j].x) minX = gtPoints[j].x;
		if (minY > gtPoints[j].y) minY = gtPoints[j].y;
	}
	Rect2f gtBB(minX, minY, maxX - minX, maxY - minY);
	return gtBB;
}

Rect2f anno2rect(vector<Point2f> annoBB)
{
    Rect2f rectBB;
    rectBB.x = min(annoBB[0].x, annoBB[1].x);
    rectBB.y = min(annoBB[0].y, annoBB[2].y);
    rectBB.width = fabs(annoBB[0].x - annoBB[1].x);
    rectBB.height = fabs(annoBB[0].y - annoBB[2].y);

    return rectBB;
}

int main()
{
	
	String modelTxt = "goturn.prototxt";
	String modelBin = "goturn.caffemodel";

	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	Net<float> net("goturn.prototxt", caffe::TEST);
	net.CopyTrainedLayersFrom("goturn.caffemodel");

	Ptr<cv::datasets::TRACK_alov> dataset = TRACK_alov::create();
	dataset->load("/opt/projects/ALOV300++");

	printf("Datasets number: %d\n", dataset->getDatasetsNum());
	for (int i = 1; i <= dataset->getDatasetsNum(); i++)
		printf("\tDataset #%d size: %d\n", i, dataset->getDatasetLength(i));

	int datasetID = 1;

	Mat prevFrame, curFrame, searchPatch, targetPatch;
	Rect2f currBB, gtBB, prevBB;
	VideoWriter outputVideo;
	double timeStamp[100];
	for (int i = 0; i < dataset->getDatasetLength(datasetID); i++)
	{
		prevFrame = curFrame.clone();
		prevBB = currBB;
		dataset->getFrame(curFrame, datasetID, i+1);

		//Draw Ground Truth BB
		Rect2f gtBB = anno2rect(dataset->getGT(datasetID, i+1));

		if (i == 0)
		{
			currBB = gtBB;
			if (gtBB.x == 0) cout << "X=0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

			//Define the codec and create VideoWriter object
			int width = curFrame.cols; // Declare width here
			int height = curFrame.rows; // Declare height here
			Size S = Size(width, height); // Declare Size structure

			// Open up the video for writing
			const string filename = "video.avi"; // Declare name of file here

			// Declare FourCC code
			int fourcc = CV_FOURCC('X','V','I','D');

			// Declare FPS here
			int fps = 5;
			outputVideo.open(filename, fourcc, fps, S);
		}
		else
		{
			timeStamp[0] = getTickCount();

			float padTarget = 2.0;
			float padSearch = 2.0;
			Rect2f searchPatchRect, targetPatchRect;
			Point2f currCenter, prevCenter;
			Mat prevFramePadded, curFramePadded;

			prevCenter.x = prevBB.x + prevBB.width / 2;
			prevCenter.y = prevBB.y + prevBB.height / 2;

			targetPatchRect.width = (float)(prevBB.width*padTarget);
			targetPatchRect.height = (float)(prevBB.height*padTarget);
			targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTarget / 2.0 + targetPatchRect.width);
			targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTarget / 2.0 + targetPatchRect.height);

			copyMakeBorder(prevFrame, prevFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
			targetPatch = prevFramePadded(targetPatchRect).clone();

			copyMakeBorder(curFrame, curFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
			searchPatch = curFramePadded(targetPatchRect).clone();

			resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
			resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

			//imshow("target", targetPatch);
			//imshow("search", searchPatch);

			//Preprocess
			//Resize
			resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
			resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

			//Mean Subtract
			targetPatch = targetPatch - 128;
			searchPatch = searchPatch - 128;

			//Convert to Float type
			targetPatch.convertTo(targetPatch, CV_32FC1);
			searchPatch.convertTo(searchPatch, CV_32FC1);

			//Wrap to Caffe Blob format
			int fullVolume = 1 * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE;
			float* data1 = new float[fullVolume];
			float* data2 = new float[fullVolume];

			int width, height;
			width = INPUT_SIZE;
			height = INPUT_SIZE;

			float* pointer;
			int offset = 0;

			vector <Mat> targetPatchSplitted;
			pointer = data1 + offset;
			for (int j = 0; j < 3; ++j)
			{
				Mat channel(height, width, CV_32FC1, pointer);
				targetPatchSplitted.push_back(channel);
				pointer += width * height;
			}

			vector <Mat> searchPatchSplitted;
			pointer = data2 + offset;
			for (int j = 0; j < 3; ++j)
			{
				Mat channel(height, width, CV_32FC1, pointer);
				searchPatchSplitted.push_back(channel);
				pointer += width * height;
			}

			split(targetPatch, targetPatchSplitted);
			split(searchPatch, searchPatchSplitted);

			//cout << targetBlob.dims() << endl;
			//cout << targetBlob.shape(0) << " " << targetBlob.shape(1) << " " << targetBlob.shape(2) << " " << targetBlob.shape(3);

			MemoryDataLayer<float> *dataLayer1 = (MemoryDataLayer<float> *) (net.layer_by_name("data1").get());
			MemoryDataLayer<float> *dataLayer2 = (MemoryDataLayer<float> *) (net.layer_by_name("data2").get());

			dataLayer1->Reset(data1, data1, 1);
			dataLayer2->Reset(data2, data2, 1);

			timeStamp[1] = getTickCount();

			net.Forward();

			timeStamp[2] = getTickCount();


			vector <Blob<float>*> outputs = net.output_blobs();
			Blob<float>* results = outputs[2];
			const float* res = results->cpu_data();

			currBB.x = targetPatchRect.x + (res[0] * targetPatchRect.width / INPUT_SIZE) - targetPatchRect.width;
			currBB.y = targetPatchRect.y + (res[1] * targetPatchRect.height / INPUT_SIZE) - targetPatchRect.height;
			currBB.width = (res[2] - res[0]) * targetPatchRect.width / INPUT_SIZE;
			currBB.height = (res[3] - res[1]) * targetPatchRect.height / INPUT_SIZE;

			cout << res[0] << " " << res[1] << " " << res[2] << " " << res[3] << endl;
			cout << currBB.x << " " << currBB.y << " " << currBB.width << " " << currBB.height << endl;
			cout << endl;

			timeStamp[3] = getTickCount();
			for (int i=0; i<3; i++)
				cout << (timeStamp[i+1] - timeStamp[i])/getTickFrequency() << "ms" << endl;

		}

		rectangle(curFrame, currBB, Scalar(0, 0, 255));
		if (gtBB.x != 0)
			rectangle(curFrame, gtBB, Scalar(0, 255, 0));
		//imshow("VOT 2015 DATASET TEST...", curFrame);
		outputVideo.write(curFrame);
		waitKey(1);


	}
	cout << "Press any button to exit";
	getchar();
	return 1;
}
