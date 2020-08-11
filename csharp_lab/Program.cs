using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using mnist;

namespace onnx_csharp_sandbox
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Starting Session");
            createSession(Int32.Parse(args[0]));
            Console.WriteLine("Session Complete");
        }


        static void createSession(int imageIndex)
        {
            string modelPath = Directory.GetCurrentDirectory() + @"/pytorch_mnist.onnx";

            // Optional : Create session options and set the graph optimization level for the session
            //SessionOptions options = new SessionOptions();
            //options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            //using (var session = new InferenceSession(modelPath, options))

            using (var session = new InferenceSession(modelPath))
            {
                //float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                Utilities.LoadTensorData();
                float[] inputData = Utilities.ImageData[imageIndex];
                string label = Utilities.ImageLabels[imageIndex];
                Console.WriteLine("Selected image is the number: " + label);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                //PrintInputMetadata(inputMeta);

                foreach (var name in inputMeta.Keys)
                {
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                using (var results = session.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    // Get the results
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output Name: {0}", r.Name);
                        int prediction = MaxProbability(r.AsTensor<float>()); 
                        Console.WriteLine("Prediction: " + prediction.ToString());
                        //Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
            }
        }

        static int MaxProbability(Tensor<float> probabilities)
        {
            float max = -9999.9F;
            int maxIndex = -1;
            for (int i = 0; i < probabilities.Length; ++i)
            {
                float prob = probabilities.GetValue(i);
                if (prob > max)
                {
                    max = prob;
                    maxIndex = i;
                }
            }
            return maxIndex;

        }

        static void PrintInputMetadata(IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            foreach (var name in inputMeta.Keys)
            {
                Console.WriteLine(name);
                Console.WriteLine("Dimension Length: " + inputMeta[name].Dimensions.Length);                    
                for (int i = 0; i < inputMeta[name].Dimensions.Length; ++i)
                {
                    Console.WriteLine(inputMeta[name].Dimensions[i]);
                }
                Console.WriteLine(inputMeta[name].ElementType.ToString());
                Console.WriteLine(inputMeta[name].IsTensor.ToString());
                Console.WriteLine(inputMeta[name].OnnxValueType.ToString());
                Console.WriteLine(inputMeta[name].SymbolicDimensions.ToString());
            }

        }

    }
}