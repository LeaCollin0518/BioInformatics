import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class Driver
{
	public static String PATH_TO_TRAINING_DATA = "/home/leac/Documents/U4/Comp401/TrainingData.arff";
	public static String PATH_TO_TESTING_DATA = "/home/leac/Documents/U4/Comp401/TestingData.arff";

	public static void main(String[] args) throws Exception
	{
		// weka classifier, J48 classification algorithm
		Classifier j48_classifier = new J48();
		
		// .arff file reader
		ArffLoader arffLoader = new ArffLoader();

		// load training data .arff file into weka objects
		File inputFile = new File(PATH_TO_TRAINING_DATA);
		arffLoader.setFile(inputFile);
		Instances instancesTrain = arffLoader.getDataSet();

		
		// load testing data .arff file into weka objects
		inputFile = new File(PATH_TO_TESTING_DATA);
		arffLoader.setFile(inputFile);
		Instances instancesTest = arffLoader.getDataSet();

		// set class index and train classifier on training data
		instancesTrain.setClassIndex(instancesTest.classIndex());
		j48_classifier.buildClassifier(instancesTrain);

		//set the Class (what we want to predict) to be Stage
		instancesTest.setClass(instancesTest.attribute("Stage"));
		
		/**
		 * TESTING THE ALGORITHM
		 * 
		 * Run the classifier on each row of the test data.
		 * For each row: print the test data, the predicted classification, and the correct classification.
		 */
		double numInstances = instancesTest.numInstances(); // how many rows of input data to test
		double correct = 0.0f;								// proportion of correct classifications
		double predicted;									// predicted classification for a given row of input data
		double actual;										// correct classification for a given row of input data

		for( int i = 0; i < numInstances; i++ )
		{
			actual    = instancesTest.instance(i).classValue() + 1;
			predicted = j48_classifier.classifyInstance(instancesTest.instance(i)) + 1;
			
			System.out.print("Instance: "        + (i + 1) + 
					       "\tActual: Stage "    + (int)(actual) +
						   "\tPredicted: Stage " + (int)predicted);
			
			// if predicted correctly on current row of test data
			if( predicted == actual )
			{
				correct++;
				System.out.println("\tCorrect");
			}
			else
			{
				System.out.println("\tIncorrect");
			}
		}
		
		// so how did we do (in percent)?
		System.out.println("\nJ48 classification precision: " + ((correct / numInstances) * 100) + "%");
	}
	
}
