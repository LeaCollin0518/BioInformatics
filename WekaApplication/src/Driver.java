import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;

public class Driver {

	public static Instances fileReader(String input) throws IOException {

		File inputFile = new File(input);
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances data = atf.getDataSet();
		
		return data;
		
	}
	
	public static void predict(Instances train, Instances test, String classifierName, 
			String [] options, String classAttribute) throws Exception {
		
		//set the Class (what we want to predict)
		test.setClass(test.attribute(classAttribute));
		
		//setting the train class index to be the same as the testing class index
		train.setClassIndex(test.classIndex());
		
		double numInst = test.numInstances(), correct = 0.0f;
		
		Classifier m_classifier = AbstractClassifier.forName(classifierName, options);
		
		//building the model
		m_classifier.buildClassifier(train);
		
		String outputFile = "/home/leac/Documents/U4/Comp401/" + classifierName + "Model.csv";
		
		//writing the header to the output csv
		File output = new File(outputFile);
		PrintWriter pw = new PrintWriter(output);
        StringBuilder sb = new StringBuilder();
        sb.append("Instance");
        sb.append(',');
        sb.append("Actual");
        sb.append(',');
        sb.append("Predicted");
        sb.append('\n');
		
		for(int i = 0;i < numInst; i++){
			
			Instance current = test.instance(i);
			
			Instance temp = (Instance)current.copy();
			
			//attributes are given as array positions, getting the string value
			String actualVal = current.stringValue(test.classIndex());
			
			//getting the predicted value of the class attribute of this instance
			double predicted = m_classifier.classifyInstance(test.instance(i));
			
			//setting this value to the temp class attribute
			temp.setValue(test.classIndex(), predicted);
			
			//getting the string value
			String predictedVal = temp.stringValue(temp.classIndex());
			
			sb.append((i+1));
			sb.append(',');
			sb.append(actualVal);
			sb.append(',');
			sb.append(predictedVal);
			sb.append('\n');
			
			
			if(predictedVal.equals(actualVal)) {
		
				correct++;
			}
		}
		
		sb.append('\n');
        sb.append("J48 classification precision: " + (100*correct/numInst) + "%");
		pw.write(sb.toString());
        pw.close();
	}
	
	public static void main(String [] args) throws Exception {
		
		//make these strings be taken in as program arguments
		String trainingFile = "/home/leac/Documents/U4/Comp401/TrainingData.arff";
		
		String testingFile = "/home/leac/Documents/U4/Comp401/TestingData.arff";
		
		String classAttribute = "Stage";
		
		String [] options = null;
		
		//reading the files and getting all the instances of each one
		Instances instancesTrain = fileReader(trainingFile);
		Instances instancesTest = fileReader(testingFile);
		
		predict(instancesTrain, instancesTest, "J48", options, classAttribute);
		predict(instancesTrain, instancesTest, "ZeroR", options, classAttribute);
		
		
	}
}
