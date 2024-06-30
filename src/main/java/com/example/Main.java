package com.example;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.CpuBindMode;
import com.mindspore.config.DataType;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;
import com.mindspore.config.Version;

public class Main {
    private static Model model;

    public static float[] generateArray(int len) {
        Random rand = new Random();
        float[] arr = new float[len];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = rand.nextFloat();
        }
        return arr;
    }

    private static ByteBuffer floatArrayToByteBuffer(float[] floats) {
        if (floats == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(floats.length * Float.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer;
    }

    private static boolean compile(String modelPath) {
        MSContext context = new MSContext();
        // use default param init context
        context.init();context.init(2, CpuBindMode.MID_CPU);
        boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        if (!ret) {
            System.err.println("Compile graph failed 1");
            context.free();
            return false;
        }
        // Create the MindSpore lite session.
        model = new Model();
        // Compile graph.
            // Set Model Type as ONNX.
        ret = model.build(modelPath, ModelType.MT_MINDIR_LITE, context);
        if (!ret) {
            System.err.println("Compile graph failed 2");
            model.free();
            return false;
        }
        return true;
    }

    private static boolean run(String imagePath) {
        // MSTensor inputTensor = model.getInputByTensorName("graph_input-173");
        // if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
        //     System.err.println("Input tensor data type is not float, the data type is " + inputTensor.getDataType());
        //     return false;
        // }
        MSTensor inputTensor = model.getInputs().get(0);
        // Generator Random Data.
        // int elementNums = inputTensor.elementsNum();
        // float[] randomData = generateArray(elementNums);
        // ByteBuffer inputData = floatArrayToByteBuffer(randomData);
        ByteBuffer inputData = null;
        try {
            inputData = processImage(imagePath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Check the input data.
        if (inputData == null) {
            System.err.println("inputData is null");
            model.free();
            return false;
        }
        // Set Input Data.
        inputTensor.setData(inputData);

        // Run Inference.
        boolean ret = model.predict();
        if (!ret) {
            inputTensor.free();
            System.err.println("MindSpore Lite run failed. 3");
            return false;
        }

        // Get Output Tensor Data.
        // MSTensor outTensor = model.getOutputByTensorName("Softmax-65");
        List<MSTensor> outTensors = model.getOutputs();
        MSTensor outTensor = outTensors.get(0);

        // Print out Tensor Data.
        StringBuilder msgSb = new StringBuilder();
        msgSb.append("out tensor shape: [");
        int[] shape = outTensor.getShape();
        for (int dim : shape) {
            msgSb.append(dim).append(",");
        }
        msgSb.append("]");
        if (outTensor.getDataType() != DataType.kNumberTypeFloat32) {
            inputTensor.free();
            outTensor.free();
            System.err.println("output tensor data type is not float, the data type is " + outTensor.getDataType());
            return false;
        }
        float[] result = outTensor.getFloatData();
        if (result == null) {
            inputTensor.free();
            outTensor.free();
            System.err.println("decodeBytes return null");
            return false;
        }
        msgSb.append(" and out data:");
        for (int i = 0; i < 50 && i < outTensor.elementsNum(); i++) {
            msgSb.append(" ").append(result[i]);
        }
        System.out.println(msgSb.toString());

        // save output image
        try {
            // System.out.println(outTensor.getDataType()); // 43, float32
            int height = shape[2];
            int width = shape[3];
            BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int[] rgbArray = new int[width * height];
            System.out.println("result length: " + result.length);
            for (int i = 0; i < height * width; i++) {
                int r = (int) (result[0 * height * width + i] * 255) & 0xFF; // Red channel
                int g = (int) (result[1 * height * width + i] * 255) & 0xFF; // Green channel
                int b = (int) (result[2 * height * width + i] * 255) & 0xFF; // Blue channel
                rgbArray[i] = (r << 16) | (g << 8) | b;
            }
            // for (int i = 0; i < height * width; i++) {
            //     int r = (int) (result[i * 3 + 0] * 255) & 0xFF; // Red channel
            //     int g = (int) (result[i * 3 + 1] * 255) & 0xFF; // Green channel
            //     int b = (int) (result[i * 3 + 2] * 255) & 0xFF; // Blue channel
            //     rgbArray[i] = (r << 16) | (g << 8) | b;
            // }
            outputImage.setRGB(0, 0, width, height, rgbArray, 0, width);
            ImageIO.write(outputImage, "png", new File("/app/EDSR/demo/data/output/flower.output.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // save result to file
        File file = new File("/app/EDSR/demo/data/output/java.flower.output.txt");
        try {
            file.createNewFile();
            // write the result to file
            FileWriter writer = new FileWriter(file);
            for (int i = 0; i < result.length-1; i++) {
                writer.write(result[i] + ",");
            }
            writer.write(result[result.length-1] + "\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // In/Out Tensor must free
        inputTensor.free();
        outTensor.free();
        return true;
    }

    private static void freeBuffer() {
        model.free();
    }

    public static ByteBuffer processImage(String imagePath) throws IOException {
        BufferedImage originalImage = ImageIO.read(new File(imagePath));
        BufferedImage resizedImage = resizeImage(originalImage, 200, 200);
        float[][][][] imageArray = convertToFloatArray(resizedImage);
        return convertToByteBuffer(imageArray);
    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
        // save output image
        try {
            ImageIO.write(outputImage, "png", new File("/app/EDSR/demo/data/output/flower.original.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return outputImage;
    }

    private static float[][][][] convertToFloatArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][][][] result = new float[1][height][width][3];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                result[0][y][x][0] = ((rgb >> 16) & 0xFF) / 255.0f; // Red
                result[0][y][x][1] = ((rgb >> 8) & 0xFF) / 255.0f;  // Green
                result[0][y][x][2] = (rgb & 0xFF) / 255.0f;         // Blue
            }
        }
        // print out the first 10 elements
        // for (int i = 0; i < 10; i++) {
        //     System.out.println("R: " + result[0][0][0][i] + ", G: " + result[0][1][0][i] + ", B: " + result[0][2][0][i]);
        // }
        return result;
    }

    private static ByteBuffer convertToByteBuffer(float[][][][] array) {
        int size = array[0].length * array[0][0].length * 3 * 4; // 4 bytes per float
        ByteBuffer buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());

        for (int h = 0; h < array[0].length; h++) {
            for (int w = 0; w < array[0][0].length; w++) {
                for (int c = 0; c < 3; c++) {
                    buffer.putFloat(array[0][h][w][c]);
                }
            }
        }

        buffer.flip();
        return buffer;
    }

    public static void main(String[] args) {
        System.out.println(Version.version());
        // if (args.length < 1) {
        //     System.err.println("The model path parameter must be passed.");
        //     return;
        // }
        // String modelPath = args[0];
        String modelPath = "/app/EDSR/demo/model/edsr_x2.ms";
        String imagePath = "/app/EDSR/demo/data/input/flower.png";
        boolean ret = compile(modelPath);
        if (!ret) {
            System.err.println("MindSpore Lite compile failed. 4");
            return;
        }

        ret = run(imagePath);
        if (!ret) {
            System.err.println("MindSpore Lite run failed. 5");
            freeBuffer();
            return;
        }

        freeBuffer();
    }
}