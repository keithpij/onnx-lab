/*
import ai.onnxruntime.OnnxMl.TensorProto;
import ai.onnxruntime.OnnxMl.TensorProto.DataType;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
*/
package src;

import src.MnistUtilities;
import src.MnistImage;

public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("Hello ONNX Runtime!");

        String currentDirectory = System.getProperty("user.dir");
        System.out.println("The current working directory is " + currentDirectory);

        String imagesFile = "./test_artifact/src/main/java/klp/t10k-images-idx3-ubyte";
        String labelsFile = "./test_artifact/src/main/java/klp/t10k-labels-idx1-ubyte";
        System.out.println(imagesFile);
        System.out.println(labelsFile);

        MnistUtilities mnist = new MnistUtilities();
        MnistImage[] data = mnist.readData(imagesFile, labelsFile);
        printMnistMatrix(data[0]);

        //Utilities.LoadTensorData();
        //String modelPath = "pytorch_mnist.onnx";

        /*        
        try (OrtSession session = env.createSession(modelPath, options)) {
           Map<String, NodeInfo> inputMetaMap = session.getInputInfo();
           Map<String, OnnxTensor> container = new HashMap<>();
           NodeInfo inputMeta = inputMetaMap.values().iterator().next();
        
           float[] inputData = Utilities.ImageData[imageIndex];
           string label = Utilities.ImageLabels[imageIndex];
           System.out.println("Selected image is the number: " + label);
        
           // this is the data for only one input tensor for this model
           Object tensorData =
                    OrtUtil.reshape(inputData, ((TensorInfo) inputMeta.getInfo()).getShape());
           OnnxTensor inputTensor = OnnxTensor.createTensor(env, tensorData);
           container.put(inputMeta.getName(), inputTensor);
        
            // Run the inference
            try (OrtSession.Result results = session.run(container)) {

                // Only iterates once
                for (Map.Entry<String, OnnxValue> r : results) {
                    OnnxValue resultValue = r.getValue();
                    OnnxTensor resultTensor = (OnnxTensor) resultValue;
                    resultTensor.getValue()
                    System.out.println("Output Name: {0}", r.Name);
                    int prediction = MaxProbability(resultTensor);
                    System.out.println("Prediction: " + prediction.ToString());
                }
            }
      
        }
        */

    }

}
