/*
 * @author Lea Collin
 */
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.converters.ArffLoader;
import weka.attributeSelection.*;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.HashMap;

public class Driver {
	
	public static void main(String [] args) throws Exception {
		
		//make these strings be taken in as program arguments
		String trainingFile = args[0];
		
		String testingFile = args[1];
		
		String outputDir = args[2];
		
		//reading the files and getting all the instances of each one
		Instances instancesTrain = fileReader(trainingFile);
		Instances instancesTest = fileReader(testingFile);
		
		//selecting most relevant attributes
		Instances [] reduced = attributeSelector(instancesTrain, instancesTest);
		
		instancesTrain = reduced[0];
		
		instancesTest = reduced[1];
		
		//what attribute do we want to predict
		String classAttribute = "Stage";
		
		String [] options = null;
		
		String [] classifiers = {"ZeroR", "J48", "RandomForest", "RandomTree", "NaiveBayes"};
		
		//need to keep track of the precision of different algorithms
		HashMap <String, Double> precisionVals = new HashMap <String, Double>();
		double maxPrecision = 0.0;
		String bestMethod = "";
		
		//running each different classifier, population HashMap to store each precision value
		for(int i = 0; i < classifiers.length; i++) {
			String toTest = classifiers[i];
			precisionVals.put(toTest, predict(instancesTrain, instancesTest, toTest, options, classAttribute, outputDir));
		}
		
		for(HashMap.Entry <String, Double> entry : precisionVals.entrySet()) {
			
			if(entry.getValue() > maxPrecision) {
				maxPrecision = entry.getValue();
				bestMethod = entry.getKey();
			}
		}
		
		System.out.println("Best Method: " + bestMethod + ", Precision: " + maxPrecision);
	}
		
	public static Instances fileReader(String input) throws IOException {

		File inputFile = new File(input);
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances data = atf.getDataSet();
		
		return data;
		
	}
	
	public static Double predict(Instances train, Instances test, String classifierName, 
			String [] options, String classAttribute, String outputDir) throws Exception {
		
		//set the Class (what we want to predict)
		test.setClass(test.attribute(classAttribute));
		
		//setting the train class index to be the same as the testing class index
		train.setClassIndex(test.classIndex());
		
		double numInst = test.numInstances(), correct = 0.0f;
		
		Classifier m_classifier = AbstractClassifier.forName(classifierName, options);
		
		//building the model
		m_classifier.buildClassifier(train);
		
		
		String outputFile = outputDir + classifierName + "Predicted.csv";
		
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
		pw.write(sb.toString());
        pw.close();
        
        Double precision = 100*correct/numInst;
        
        return precision;
	}
	
	public static Instances [] attributeSelector(Instances train, Instances test) throws Exception {
		
		AttributeSelection selector = new AttributeSelection();
		
		//CfsSubsetEval evaluator = new CfsSubsetEval();
		//BestFirst search = new BestFirst();
		
		selector.SelectAttributes(train);
		
		Instances trainTemp = selector.reduceDimensionality(train);
		Instances trainTest = selector.reduceDimensionality(test);
		
		Instances [] reduced = {trainTemp, trainTest};

		return reduced;
	}
}
