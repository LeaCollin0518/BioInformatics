import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffLoader.ArffReader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class Driver {

	private static final Attribute Attribute = null;

	public static Instances fileReader(String input) throws IOException {

		BufferedReader reader = new BufferedReader(new FileReader(input));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		
		return data;
		
	}
	
	public static void main(String [] args) throws Exception {
		String trainingFile = "/home/leac/Documents/U4/Comp401/TrainingData.arff";
		//Instances trainingData = fileReader(trainingFile);
		
		String testingFile = "/home/leac/Documents/U4/Comp401/TestingData.arff";
		//Instances testingData = fileReader(testingFile);
		
		Classifier m_classifier = new J48();
		
		
		File inputFile = new File(trainingFile);	//creating a new file (training file)
		ArffLoader atf = new ArffLoader();	//creating a new ArffLoader
		atf.setFile(inputFile);		//setting the file for the ArffLoader
		Instances instancesTrain = atf.getDataSet(); 	//reading the entire dataset

		//same thing as right above, with testing file
		inputFile = new File(testingFile);
		atf.setFile(inputFile);
		Instances instancesTest = atf.getDataSet();
		
		instancesTest.setClass(instancesTest.attribute("Stage")); //set the Class (what we want to predict) to be Stage
		
		instancesTrain.setClassIndex(instancesTest.classIndex()); //set the ClassIndex to be the classIndex (straightforward)
		
		double sum = instancesTest.numInstances(), correct = 0.0f;
				
		m_classifier.buildClassifier(instancesTrain);
		
		for(int i = 0;i<sum;i++){
			
			Instance current = instancesTest.instance(i);
			
			Instance temp = (Instance)current.copy();
			
			String actualVal = current.stringValue(instancesTest.classIndex());
			
			double predicted = m_classifier.classifyInstance(instancesTest.instance(i));
			
			temp.setValue(instancesTest.classIndex(), predicted);
			
			String predictedVal = temp.stringValue(temp.classIndex());
			
			
			System.out.print("Instance: " + (i+1) + 
					"		Actual: " + actualVal + 
					"		Predicted: " + predictedVal);
			
			
			if(predictedVal.equals(actualVal)) {// If the prediction of value and value are equal (classified in the testing corpus provides must be the correct answer, the results are meaningful)
		
				correct++; //The correct value 1
				System.out.println("		Correct");
			}
			else {
				System.out.println("		Incorrect");
			}
		}
		
		System.out.println();
		System.out.println("J48 classification precision: " + (100*correct/sum) + "%");
	}
	
}
