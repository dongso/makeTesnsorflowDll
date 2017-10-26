#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include "stdlib.h"
#include "assert.h"
#include <Windows.h>

#include <stdio.h>
#include <list>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

#include "tensorJK.h"


using namespace tensorflow;
using namespace cv;
using namespace std;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


#define LOG_OUT(_x_) OutputDebugString(_x_) 
#define LOG_OUT_W(_x_)  OutputDebugStringW(_x_) 
#define LOG_OUT_A(_x_)  OutputDebugStringA(_x_) 


bool LoadBmp(char* filename, byte** pImage)
{
	FILE* fp;

	// 비트맵 정보 구조체
	BITMAPFILEHEADER BMFH;					///< BitMap File Header.
	BITMAPINFOHEADER BMIH;					///< BitMap Info Header.

	fopen_s(&fp, filename, "rb");
	if (nullptr == fp)
	{
		LOG_OUT_A("fopen() error");
		return false;
	}

	// 지정한 크기만큼 파일에서 읽어옴, 그리고 비트맵파일헤더에 넣어줌
	fread(&BMFH, sizeof(BITMAPFILEHEADER), 1, fp);
	if (BMFH.bfType != 0x4d42)	// 비트맵 파일이 아니면 종료한다.
	{
		fclose(fp);
		LOG_OUT_A("not '.bmp' file !!!");
		return false;
	}

	fread(&BMIH, sizeof(BITMAPINFOHEADER), 1, fp);	//인포헤더에 있는 크기의 정보만큼 읽어서
	if (BMIH.biBitCount != 24 || BMIH.biCompression != BI_RGB) //24비트인지 체크하고, 압축이 안되어 있는지 체크를 함
	{
		fclose(fp);
		return false;
	}

	INT Width = BMIH.biWidth;
	INT Height = BMIH.biHeight - 1;
	INT BytePerScanline = (Width * 3 + 3) & ~3;  // 패딩
	INT size = BMFH.bfSize;

	*pImage = (BYTE*)malloc(size);

	fread(*pImage, size, 1, fp);  // 파일의 정보를 전부 읽어온다.
								  //*pImage += BytePerScanline * Height;

								  // FILE*을 닫음.
	fclose(fp);

	return true;
}

bool SaveBmp(char* filename, byte* pImage, int width, int height)
{
	// DIB의 형식을 정의한다.
	BITMAPINFO dib_define;
	dib_define.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	dib_define.bmiHeader.biWidth = width;
	dib_define.bmiHeader.biHeight = height;
	dib_define.bmiHeader.biPlanes = 1;
	dib_define.bmiHeader.biBitCount = 24;
	dib_define.bmiHeader.biCompression = BI_RGB;
	dib_define.bmiHeader.biSizeImage = (((width * 24 + 31) & ~31) >> 3) * height;
	dib_define.bmiHeader.biXPelsPerMeter = 0;
	dib_define.bmiHeader.biYPelsPerMeter = 0;
	dib_define.bmiHeader.biClrImportant = 0;
	dib_define.bmiHeader.biClrUsed = 0;

	// DIB 파일의 헤더 내용을 구성한다.
	BITMAPFILEHEADER dib_format_layout;
	ZeroMemory(&dib_format_layout, sizeof(BITMAPFILEHEADER));
	dib_format_layout.bfType = *(WORD*)"BM";
	dib_format_layout.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);// +dib_define.bmiHeader.biSizeImage;
	dib_format_layout.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	// 파일 생성.
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wb");
	if (nullptr == fp)
	{
		LOG_OUT_A("fopen() error");
		return false;
	}

	// 생성 후 헤더 및 데이터 쓰기.
	fwrite(&dib_format_layout, 1, sizeof(BITMAPFILEHEADER), fp);
	fwrite(&dib_define, 1, sizeof(BITMAPINFOHEADER), fp);
	fwrite(pImage, 1, dib_define.bmiHeader.biSizeImage, fp);
	fclose(fp);

	return true;
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
	size_t* found_label_count) {
	std::ifstream file(file_name);
	if (!file) {
		return tensorflow::errors::NotFound("Labels file ", file_name,
			" not found.");
	}
	result->clear();
	string line;
	while (std::getline(file, line)) {
		result->push_back(line);
	}
	*found_label_count = result->size();
	const int padding = 16;
	while (result->size() % padding) {
		result->emplace_back();
	}
	return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
	Tensor* output) {
	tensorflow::uint64 file_size = 0;
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

	string contents;
	contents.resize(file_size);

	std::unique_ptr<tensorflow::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	tensorflow::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
	if (data.size() != file_size) {
		return tensorflow::errors::DataLoss("Truncated read of '", filename,
			"' expected ", file_size, " got ",
			data.size());
	}
	output->scalar<string>()() = data.ToString();
	return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
	const int input_width, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors)
{
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string input_name = "file_reader";
	string output_name = "normalized";

	// read file_name into a tensor named input
	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
	TF_RETURN_IF_ERROR(
		ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

	// use a placeholder to read input data
	auto file_reader =
		Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "input", input },
	};

	// Now try to figure out what kind of file it is and decode it.
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader =
			Squeeze(root.WithOpName("squeeze_first_dim"),
				DecodeGif(root.WithOpName("gif_reader"), file_reader));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
		image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
	}
	else {
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	// Now cast the image data to float so we can do normal math on it.
	auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

	// The convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims().
	auto dims_expander = ExpandDims(root, float_caster, 0);

	// Bilinearly resize the image to fit the required dimensions.
	auto resized = ResizeBilinear( 	root, dims_expander,	Const(root.WithOpName("size"), { input_height, input_width } ) );

	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }), 	{ input_std });

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { output_name }, {}, out_tensors));
	return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
	std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
	Tensor* indices, Tensor* scores) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string output_name = "top_k";
	TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensors.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	// The TopK node returns two outputs, the scores and their original indices,
	// so we have to append :0 and :1 to specify them both.
	std::vector<Tensor> out_tensors;
	TF_RETURN_IF_ERROR(session->Run({}, { output_name + ":0", output_name + ":1" },
	{}, &out_tensors));
	*scores = out_tensors[0];
	*indices = out_tensors[1];
	return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
	const string& labels_file_name) {
	std::vector<string> labels;
	size_t label_count;
	Status read_labels_status =
		ReadLabelsFile(labels_file_name, &labels, &label_count);
	if (!read_labels_status.ok()) {
		LOG(ERROR) << read_labels_status;
		return read_labels_status;
	}
	const int how_many_labels = std::min(5, static_cast<int>(label_count));
	Tensor indices;
	Tensor scores;
	TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
	tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	for (int pos = 0; pos < how_many_labels; ++pos) {
		const int label_index = indices_flat(pos);
		const float score = scores_flat(pos);
		LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
	}
	return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
	bool* is_expected) {
	*is_expected = false;
	Tensor indices;
	Tensor scores;
	const int how_many_labels = 1;
	TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	if (indices_flat(0) != expected) {
		LOG(ERROR) << "Expected label #" << expected << " but got #"
			<< indices_flat(0);
		*is_expected = false;
	}
	else {
		*is_expected = true;
	}
	return Status::OK();
}

int test(int height, int width, byte* pImage)
{
	cv::Mat cameraImg(height, width, CV_8UC3, pImage);	// CV_8UC3  CV_32FC4

	std::cout << "==================================" << std::endl;
	Mat imagePixels;
	cameraImg.convertTo(imagePixels, CV_32FC3);
	cv::imshow("aa.png", cameraImg);
	std::cout << "==================================" << std::endl;
	return 10;
}




int ReleaseMemory(int* pArray)
{
	delete[] pArray;
	return 0;
}

float* runCNN_test(int height, int width, unsigned char* pImage)
{	
	//cv::Mat cameraImg(height, width, CV_8UC3, pImage);	// CV_8UC3  CV_32FC4
	cv::Mat cameraImg = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/gs.jpg", IMREAD_COLOR);
					// = imread("D:/JKfactory/my_projects/color_defect_galaxy/programs/galaxy_color_classification_CNN_JK/data/test/c.png", IMREAD_COLOR);

	//===== connect Mat with Tensor =====//
	Tensor inputImg(DT_FLOAT, TensorShape({ 1,height, width, 4 }));
	float *p = inputImg.flat<float>().data();
	cv::Mat bufferImg(height, width, CV_32FC4, p);
			
	Mat imagePixels;
	cameraImg.convertTo(imagePixels, CV_32FC4, 1/255.0);// 
	//cameraImg2.copyTo(imagePixels);

	cv::imshow("imagePixels", imagePixels);
	cv::waitKey(0);

	Mat newImage(32, 32, CV_32FC4, cv::Scalar(100));
	Mat scatter(32, 32, CV_32FC1, cv::Scalar(0.9));
	
	Mat img(32, 32, CV_8UC3); // declare three channels image 
	Mat ch1, ch2, ch3; // declare three matrices 
					   // "channels" is a vector of 3 Mat arrays:
	vector<Mat> channels(4);
	// split img:
	split(imagePixels, channels);
	// get the channels (follow BGR order in OpenCV)
	ch1 = channels[0];  
	ch2 = channels[1];
	ch3 = channels[2];

	// modify channel// then merge
	merge(channels, img);	
	
	cv::imshow("img", img);
	cv::waitKey(0);

	
	channels.push_back(scatter);
	merge(channels, newImage);
	newImage.convertTo(bufferImg, CV_32FC4);
	std::cout << inputImg.tensor<float, 4>() << std::endl;

//
//	Session* session;
//	Status status = NewSession(SessionOptions(), &session);
//	if (!status.ok()) {
//		std::cout << status.ToString() << "\n";
//		//return 1;
//	}
//
//	// Read in the protobuf graph we exported
//	// (The path seems to be relative to the cwd. Keep this in mind
//	// when using `bazel run` since the cwd isn't where you call
//	// `bazel run` but from inside a temp folder.)
//	GraphDef graph_def;
//	//status = ReadBinaryProto(Env::Default(), "C:/Users/sjk07/Desktop/genFrozenGraph/models/output_graph.pb", &graph_def);
//	//status = ReadTextProto(Env::Default(), "C:/Users/sjk07/Desktop/genFrozenGraph/models/input_graph.pb", &graph_def);
//	status = ReadBinaryProto(Env::Default(), "D:/JKfactory/my_projects/color_defect_galaxy/programs/galaxy_color_classification_CNN_JK/galaxy8A_Lab/output_graph.pb", &graph_def);
//	//status = ReadTextProto(Env::Default(), "C:/Users/sjk07/Desktop/genFrozenGraph/models/input_graph.pb", &graph_def);
//
//	if (!status.ok()) {
//		std::cout << status.ToString() << "\n";
//		//return 1;
//	}
//
//	// Add the graph to the session  
//	status = session->Create(graph_def);
//	if (!status.ok()) {
//		std::cout << status.ToString() << "\n";
//		//return 1;
//	}
//
//	// Setup inputs and outputs:
//
//	// Our graph doesn't require any inputs, since it specifies default values,
//	// but we'll change an input to demonstrate.
//	Tensor training(DT_BOOL, TensorShape());
//	training.scalar<bool>()() = false;
//
//
//	/*Tensor image(DT_FLOAT, TensorShape());
//	image.scalar<float>()() = 1.0;*/
//
//	//string image = "D:/JKfactory/my_projects/color_defect_galaxy/programs/galaxy_color_classification_CNN_JK/data/test/c.png";
//	//string image2 = "D:/JKfactory/my_projects/color_defect_galaxy/programs/galaxy_color_classification_CNN_JK/data/test/c.png";
//	//int32 input_width = 32;
//	//int32 input_height = 32;
//	//float input_mean = 0;
//	//float input_std = 255;
//
//	//bool self_test = false;
//	//string root_dir = "";
//
//	//// Get the image from disk as a float array of numbers, resized and normalized
//	//// to the specifications the main graph expects.
//	//std::vector<Tensor> resized_tensors;
//	//string image_path = tensorflow::io::JoinPath(root_dir, image);
//	//Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width, input_mean, input_std, &resized_tensors);
//
//	//if (!read_tensor_status.ok()) {
//	//	LOG(ERROR) << read_tensor_status;
//	//	return -1;
//	//}
//	//Tensor& resized_tensor1 = resized_tensors[0];
//
//	//std::vector<Tensor> resized_tensors2;
//	//string image_path2 = tensorflow::io::JoinPath(root_dir, image2);
//	//Status read_tensor_status2 = ReadTensorFromImageFile(image_path2, input_height, input_width, input_mean, input_std, &resized_tensors2);
//	//Tensor& resized_tensor2 = resized_tensors2[0];
//
//	std::vector<Tensor> resized_tensors_final;
//	resized_tensors_final.push_back(inputImg);
//	//resized_tensors_final.push_back(resized_tensor2);
//
//	std::vector<Tensor> logits;
//
//	//for (int i = 0; i<1; i++)
//	//{
//	Tensor& resized_tensor = resized_tensors_final[0];
//
//	//std::cout << resized_tensor.tensor<float, 4>() << std::endl;
//
//	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
//		{ "galaxy/training", training },
//		{ "galaxy/input", resized_tensor },  //inputImg
//	};
//
//	// The session will initialize the outputs
//	std::vector<tensorflow::Tensor> outputs;
//
//	// Run the session, evaluating our "c" operation from the graph
//	status = session->Run(inputs, { "prob" }, {}, &outputs);
//	if (!status.ok()) {
//		std::cout << status.ToString() << "\n";
////		return 1;
//	}
//
//	// Grab the first output (we only evaluated one graph node: "c")
//	// and convert the node to a scalar representation.
//
//	const Tensor& logit = outputs[0];
//	auto logit_softmax = logit.tensor<float, 2>();
//
//	logits.push_back(logit);
//	//auto output_c = outputs[0].scalar<float>();
//
//	// (There are similar methods for vectors and matrices here:
//	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)
//
//	// Print the results
//	//std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
//
//	// Free any resources used by the session
//	//session->Close();
//	//std::cout << logit_softmax << "\n";
//	//}
//	
//
//	int nSamples = 2;
//	float* results = new float[nSamples];
//	for (int i = 0; i < nSamples; i++)
//	{
//		results[i] = logit_softmax(i);
//	}
//	
//	//results = logit_softmax.data<1, 2>();
//
	return 0;
}



float main()
{
	//===== connect Mat with Tensor =====//
	int inputHeight = 300;
	int inputWidth = 1000;
	int nx = 10;
	int ny = 3;

	Mat imgMat = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/S8_A_02NG_01_L190.bmp", IMREAD_COLOR);
	//cv::imshow("S8_A_02NG_01_L190.bmp", imgMat);
	//cv::waitKey(0);
	//std::cout << imgMat.cols << ' | ' <<  imgMat.rows << std::endl;

	//runCNN_test(imgMat.cols, imgMat.rows, imgMat.ptr());
	
	runCnnBlockC4( nx, ny, imgMat.cols, imgMat.rows, imgMat.ptr(), 15000);
	//test(imgMat.rows, imgMat.cols, imgMat.ptr());

	return 0;
}

float* runCnnFullC3(int nx, int ny, int width, int height, unsigned char* pImage, int scatter)
{
	// declare a session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// read binary frozen graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "output_graph.pb", &graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// Add the graph to the session  
	status = session->Create(graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/


	//cv::Mat cameraImage(height, width, CV_8UC3, pImage);		// C# image pointer to C++ Mat
	cv::Mat cameraImage = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/S8_A_02NG_01_L190.bmp", IMREAD_COLOR);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2RGB);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2Lab);

	cameraImage.convertTo(cameraImage, CV_32FC3, 1 / 255.0);	// 8uint-3channel  ->  32float-4channel

	/*cv::imshow("cameraImage", cameraImage);
	cv::waitKey(0);*/

	// set inpute variable
	Tensor training(DT_BOOL, TensorShape());
	training.scalar<bool>()() = false;

	int labelSize = 2;

	float* results = new float[labelSize];
	//std::vector<Tensor> logits;

	// connect a Mat sampleImage with tensor inputSample
	Tensor inputSample(DT_FLOAT, TensorShape({ 1, height, width, 3 }));
	float *ptrTensor = inputSample.flat<float>().data();
	cv::Mat bufferImg(height, width, CV_32FC3, ptrTensor);
	cameraImage.convertTo(bufferImg, CV_32FC3);		//
															//std::cout << inputSample.tensor<float, 4>() << std::endl;

															// set inputs
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "G8A/training", training },
		{ "G8A/input", inputSample },  //inputImgzz
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "prob" }, {}, &outputs);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	const Tensor& logit = outputs[0];
	auto logit_softmax = logit.tensor<float, 2>();
	for (int i = 0; i < labelSize; i++)
	{
		results[i] = logit_softmax(i);
		std::cout << results[i] << std::endl;
	}

	return results;
}


float* runCnnFullC4(int nx, int ny, int width, int height, unsigned char* pImage, int scatter)
{
	// declare a session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// read binary frozen graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "output_graph.pb", &graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// Add the graph to the session  
	status = session->Create(graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/


	//cv::Mat cameraImage(height, width, CV_8UC3, pImage);		// C# image pointer to C++ Mat
	cv::Mat cameraImage = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/S8_A_02NG_01_L190.bmp", IMREAD_COLOR);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2RGB);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2Lab);

	cameraImage.convertTo(cameraImage, CV_32FC4, 1 / 255.0);	// 8uint-3channel  ->  32float-4channel

	/*cv::imshow("cameraImage", cameraImage);
	cv::waitKey(0);*/

	float maxScatter = 25000;
	Mat imageWithScatter(height, width, CV_32FC4, cv::Scalar(100));						// declare a image-4channel with scatter
	Mat scatterMat(height, width, CV_32FC1, cv::Scalar((float)scatter / maxScatter));	// declare a scatter matrix

	vector<Mat> channels(4);		// 
	split(cameraImage, channels);		// split cameraImage:
	channels.push_back(scatterMat);		// append scatter matrix
	merge(channels, imageWithScatter);	// make a image-4channel including scatter

	// set inpute variable
	Tensor training(DT_BOOL, TensorShape());
	training.scalar<bool>()() = false;

	int labelSize = 2;	
	float* results = new float[labelSize];
	//std::vector<Tensor> logits;

	// connect a Mat sampleImage with tensor inputSample
	Tensor inputSample(DT_FLOAT, TensorShape({ 1, height, width, 4 }));
	float *ptrTensor = inputSample.flat<float>().data();
	cv::Mat bufferImg(height, width, CV_32FC4, ptrTensor);
	imageWithScatter.convertTo(bufferImg, CV_32FC4);		//
	//std::cout << inputSample.tensor<float, 4>() << std::endl;

	// set inputs
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "G8A/training", training },
		{ "G8A/input", inputSample },  //inputImgzz
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "prob" }, {}, &outputs);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	const Tensor& logit = outputs[0];
	auto logit_softmax = logit.tensor<float, 2>();
	for (int i = 0; i < labelSize; i++)
	{
		results[i] = logit_softmax(i);
		std::cout << results[i] << std::endl;
	}		
	
	return results;
}

float* runCnnBlockC3(int nx, int ny, int width, int height, unsigned char* pImage)
{
	// declare a session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// read binary frozen graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "output_graph.pb", &graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/

	// Add the graph to the session  
	status = session->Create(graph_def);
	/*if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 0;
	}*/


	//cv::Mat cameraImage(height, width, CV_8UC3, pImage);		// C# image pointer to C++ Mat
	cv::Mat cameraImage = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/S8_A_02NG_01_L190.bmp", IMREAD_COLOR);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2RGB);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2Lab);

	cameraImage.convertTo(cameraImage, CV_32FC3, 1 / 255.0);	// 8uint-3channel  ->  32float-4channel

	/*cv::imshow("cameraImage", cameraImage);
	cv::waitKey(0);*/


	//Tensor test(DT_FLOAT, TensorShape({ 1, height, width, 3 }));
	//float *p = test.flat<float>().data();
	//cv::Mat bufferImg(height, width, CV_32FC4, p);
	//imageWithScatter.convertTo(bufferImg, CV_32FC4);		//
	//std::cout << test.tensor<float, 4>() << std::endl;


	// set inpute variable
	Tensor training(DT_BOOL, TensorShape());
	training.scalar<bool>()() = false;

	int featureWidth = 32;
	int featureHeight = 32;
	int margin = 5;
	int nAreas = nx*ny;
	int dx = (int)((width - 2 * margin) / nx);
	int dy = (int)((height - 2 * margin) / ny);

	float* results = new float[nAreas];
	//std::vector<Tensor> logits;

	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
		{
			int startX = margin + i*dx + (int)(0.5*(dx - featureWidth));
			int startY = margin + j*dy + (int)(0.5*(dy - featureHeight));

			// Setup a rectangle to define your region of interest
			cv::Rect myROI(startX, startY, featureWidth, featureHeight);

			// Crop the full image to that image contained by the rectangle myROI
			// Note that this doesn't copy the data
			cv::Mat croppedImage = cameraImage(myROI); // cameraImage(myROI);

															/*cv::imshow("croppedImage", croppedImage);
															cv::waitKey(0);*/

															// connect a Mat sampleImage with tensor inputSample
			Tensor inputSample(DT_FLOAT, TensorShape({ 1, featureHeight, featureWidth, 3 }));
			float *ptrTensor = inputSample.flat<float>().data();
			cv::Mat bufferImg(featureHeight, featureWidth, CV_32FC3, ptrTensor);
			croppedImage.convertTo(bufferImg, CV_32FC3);		//
																//std::cout << inputSample.tensor<float, 4>() << std::endl;

																// set inputs
			std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
				{ "G8A/training", training },
				{ "G8A/input", inputSample },  //inputImgzz
			};

			// The session will initialize the outputs
			std::vector<tensorflow::Tensor> outputs;

			// Run the session, evaluating our "c" operation from the graph
			status = session->Run(inputs, { "prob" }, {}, &outputs);
			/*if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			return 0;
			}*/

			const Tensor& logit = outputs[0];
			auto logit_softmax = logit.tensor<float, 2>();
			results[i + j*nx] = logit_softmax(0);
			//std::cout << results[i] << std::endl;
		}

	return results;
}

float* runCnnBlockC4(int nx, int ny, int width, int height, unsigned char* pImage, int scatter)
{
	// declare a session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	/*if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 0;
	}*/
		
	// read binary frozen graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "output_graph.pb", &graph_def);	
	/*if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 0;
	}*/

	// Add the graph to the session  
	status = session->Create(graph_def);
	/*if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 0;
	}*/

	
	//cv::Mat cameraImage(height, width, CV_8UC3, pImage);		// C# image pointer to C++ Mat
	cv::Mat cameraImage = imread("D:/my_projects/git_local_server/colorCheckCNN_git/makeTesnsorflowDll/x64/Release/S8_A_02NG_01_L190.bmp", IMREAD_COLOR);	
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2RGB);
	cv::cvtColor(cameraImage, cameraImage, CV_BGR2Lab);

	cameraImage.convertTo(cameraImage, CV_32FC4, 1 / 255.0);	// 8uint-3channel  ->  32float-4channel

	/*cv::imshow("cameraImage", cameraImage);
	cv::waitKey(0);*/

	float maxScatter = 25000;
	Mat imageWithScatter(height, width, CV_32FC4, cv::Scalar(100));						// declare a image-4channel with scatter
	Mat scatterMat(height, width, CV_32FC1, cv::Scalar((float)scatter/ maxScatter));	// declare a scatter matrix

	vector<Mat> channels(4);		// 
	split(cameraImage, channels);		// split cameraImage:
	channels.push_back(scatterMat);		// append scatter matrix
	merge(channels, imageWithScatter);	// make a image-4channel including scatter

	//Tensor test(DT_FLOAT, TensorShape({ 1, height, width, 4 }));
	//float *p = test.flat<float>().data();
	//cv::Mat bufferImg(height, width, CV_32FC4, p);
	//imageWithScatter.convertTo(bufferImg, CV_32FC4);		//
	//std::cout << test.tensor<float, 4>() << std::endl;

	// set inpute variable
	Tensor training(DT_BOOL, TensorShape());
	training.scalar<bool>()() = false;
	
	int featureWidth = 32;
	int featureHeight = 32;
	int margin = 5;
	int nAreas = nx*ny;	
	int dx = (int)((width - 2 * margin) / nx);
	int dy = (int)((height - 2 * margin) / ny);

	float* results = new float[nAreas];
	//std::vector<Tensor> logits;

	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
		{
			int startX = margin + i*dx + (int)(0.5*(dx - featureWidth));
			int startY = margin + j*dy + (int)(0.5*(dy - featureHeight));

			// Setup a rectangle to define your region of interest
			cv::Rect myROI(startX, startY, featureWidth, featureHeight);

			// Crop the full image to that image contained by the rectangle myROI
			// Note that this doesn't copy the data
			cv::Mat croppedImage = imageWithScatter(myROI); // cameraImage(myROI);

			/*cv::imshow("croppedImage", croppedImage); 
			cv::waitKey(0);*/

			// connect a Mat sampleImage with tensor inputSample
			Tensor inputSample(DT_FLOAT, TensorShape({ 1, featureHeight, featureWidth, 4 }));
			float *ptrTensor = inputSample.flat<float>().data();
			cv::Mat bufferImg(featureHeight, featureWidth, CV_32FC4, ptrTensor);
			croppedImage.convertTo(bufferImg, CV_32FC4);		//
			//std::cout << inputSample.tensor<float, 4>() << std::endl;

			// set inputs
			std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
				{ "G8A/training", training },
				{ "G8A/input", inputSample },  //inputImgzz
			};

			// The session will initialize the outputs
			std::vector<tensorflow::Tensor> outputs;

			// Run the session, evaluating our "c" operation from the graph
			status = session->Run(inputs, { "prob" }, {}, &outputs);
			/*if (!status.ok()) {
				std::cout << status.ToString() << "\n";
				return 0;
			}*/

			const Tensor& logit = outputs[0];
			auto logit_softmax = logit.tensor<float, 2>();
			results[i+j*nx] = logit_softmax(0);
			std::cout << results[i] << std::endl;
		}
	
	return results;
}