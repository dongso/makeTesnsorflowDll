#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include "stdlib.h"
#include "assert.h"
#include <windows.h>
#include <stdio.h>
#include <list>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>


using namespace cv;
using namespace std;

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using namespace std;

extern "C" __declspec(dllexport) float* runCnnFullC3(int nx, int ny, int width, int height, unsigned char* pImage, int scatter);
extern "C" __declspec(dllexport) float* runCnnFullC4(int nx, int ny, int width, int height, unsigned char* pImage, int scatter);
extern "C" __declspec(dllexport) float* runCnnBlockC3(int nx, int ny, int width, int height, unsigned char* pImage);
extern "C" __declspec(dllexport) float* runCnnBlockC4(int nx, int ny, int width, int height, unsigned char* pImage, int scatter);
extern "C" __declspec(dllexport) int ReleaseMemory(int* pArray);

//
//class TensorflowJK
//{
//public:
//	TensorflowJK();
//	float solve();
//
//private:
//	Session* session;
//	Status status;
//	GraphDef graph_def;
//
//
//};