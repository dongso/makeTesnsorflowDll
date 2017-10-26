
#include <vector>
#include <iostream>
#include "tensorJK.h"

using namespace std;

TensorflowJK::TensorflowJK()
{
	std::cout << "constructor" << std::endl;

	status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		//return 1;
	}

	// Read in the protobuf graph we exported
	// (The path seems to be relative to the cwd. Keep this in mind
	// when using `bazel run` since the cwd isn't where you call
	// `bazel run` but from inside a temp folder.)
	graph_def;
	//status = ReadBinaryProto(Env::Default(), "C:/Users/sjk07/Desktop/models/graph.pb", &graph_def);
	status = ReadTextProto(Env::Default(), "C:/Users/sjk07/Desktop/models/graph.pb", &graph_def);

	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		//return 1;
	}

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		//return 1;
	}

}

float TensorflowJK::solve()
{
	// Setup inputs and outputs:

	// Our graph doesn't require any inputs, since it specifies default values,
	// but we'll change an input to demonstrate.
	Tensor a(DT_FLOAT, TensorShape());
	a.scalar<float>()() = 1.0;

	Tensor b(DT_FLOAT, TensorShape());
	b.scalar<float>()() = 12.0;

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "a", a },
		{ "b", b },
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "cc" }, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Grab the first output (we only evaluated one graph node: "c")
	// and convert the node to a scalar representation.
	auto output_c = outputs[0].scalar<float>();

	// (There are similar methods for vectors and matrices here:
	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

	// Print the results
	std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>

									 // Free any resources used by the session
	//session->Close();

	return output_c();
}