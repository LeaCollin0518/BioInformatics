/*
 * @author Lea Collin
 */
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import org.postgresql.util.PSQLException;
import java.sql.*;

public class Driver {
	
	static Scanner sc = new Scanner(System.in);
	
	public static void main(String [] args) throws Exception {
		
		String dbConfig = args[0];
		
		//ArrayList will contain all the possible algorithms that the user can input
		//needs to be expanded on still probably
		ArrayList<String> possibleClassifiers = new ArrayList<String>();
		possibleClassifiers.add("ZeroR");
		possibleClassifiers.add("NaiveBayes");
		possibleClassifiers.add("Logisitc");
		possibleClassifiers.add("MultilayerPerceptron");
		possibleClassifiers.add("SimpleLogisitc");
		possibleClassifiers.add("SMO");
		possibleClassifiers.add("IBk");
		possibleClassifiers.add("KStar");
		possibleClassifiers.add("LWL");
		possibleClassifiers.add("DecisionStump");
		possibleClassifiers.add("HoeffdingTree");
		possibleClassifiers.add("J48");
		possibleClassifiers.add("LMT");
		possibleClassifiers.add("RandomForest");
		possibleClassifiers.add("RandomTree");
		possibleClassifiers.add("REPTree");
		
			
		System.out.println("A BUNCH OF DIRECTIONS");
		System.out.println("Please enter the directory name of where you would like to store all program outputs:");
		
		//add control to check for valid directory
		//String outputDir = sc.next();
		String outputDir = args[1];
		
		System.out.println();
		
		System.out.println("Please enter the name of the file you'd like to store the TRAINING data. Please end the file name in '.arff'");
		String trainingFile = setFileName(".arff", outputDir);
		System.out.println();
		
		System.out.println("Please enter the name of the file you'd like to store the TESTING data. Please end the file name in '.arff'");
		String testingFile = setFileName(".arff", outputDir);
		System.out.println();
		
		//trying to connect to database given username and password, user prompted to enter username and password again if connection is unsuccessful
		boolean successfulConnection = false;
		while(!successfulConnection) {
				System.out.println("Please enter your database username:");
				String dbUsr = sc.next();
				//String dbUsr = args[2];
				
				System.out.println("Password:");
				String dbPwd = sc.next();
				sc.nextLine();
				//String dbPwd = args[3];
			try {
				connectToDatabase(dbUsr, dbPwd, dbConfig, trainingFile, testingFile);
				
				successfulConnection = true;
			}catch (PSQLException s){
				System.out.println("Username or password was incorrect. Please try again.");
			}
		}
		
		
		//getting user input for classifier names, checking if input is valid
		String [] classifiers = null;
		
		boolean validClassifier = false;
		while(!validClassifier){
			System.out.println("Please enter the names of the classifiers you'd like to test, separated by a single space.");
			String userInput = sc.nextLine();
			
			//classifiers = {"ZeroR", "J48", "RandomTree", "RandomForest", "NaiveBayes"};
			classifiers = userInput.split("\\s+");
			for(int i = 0; i < classifiers.length; i++) {
				if(!possibleClassifiers.contains(classifiers[i])) {
					System.out.println("You entered an incorrect classifier name.");
					System.out.println("Please try again.");
					System.out.println();
					break;
				}
				
				if(i == classifiers.length - 1 && possibleClassifiers.contains(classifiers[i])) {
					validClassifier = true;
				}
			}
		}
		
		System.out.println();
		System.out.println("Please enter the name of the file you would like to store all of the predictions. Please end the file in '.csv'");
		String predictionFile = setFileName(".csv", outputDir);
		
		System.out.println();
		System.out.println("Finally, please enter the name of the file you would like to store the precision of each algorithm you are testing. "
				+ "Please end the file in '.csv'");
		String precisionFile = setFileName(".csv", outputDir);
		System.out.println();
		
		sc.close();
		
		//reading the files and getting all the instances of each one
		Instances instancesTrain = fileReader(trainingFile);
		Instances instancesTest = fileReader(testingFile);
				
		//what attribute do we want to predict
		String classAttribute = "Stage";
		

		String [] attributesToRemove = {"barcode", "das"};
		String indicesToRemove = removeAttribute(instancesTrain, attributesToRemove);
		
		//will be used to store the highest precision and most precise classifier name
		Double max = 0.0;
		String bestMethod = "";
		
		//run all the different classifiers
		Double [] precisions = predict(instancesTrain, instancesTest, classifiers, indicesToRemove, classAttribute, predictionFile);
		
		
		//writing precision values to a csv
				File output = new File(precisionFile);
				PrintWriter pw = new PrintWriter(output);
		        StringBuilder sb = new StringBuilder();
		        sb.append("Algorithm");
		        sb.append(',');
		        sb.append("Precision");
		        sb.append("\n");
		        
		//find best algorithm and write to file
		for(int i = 0; i < classifiers.length; i++) {
			System.out.println("Algorithm: " + classifiers[i] + ", Precision: " + precisions[i]);
			sb.append(classifiers[i] + ",");
			sb.append(precisions[i]);
			
			if(i != classifiers.length - 1) {
				sb.append("\n");
			}
			if(precisions[i] > max) {
				max = precisions[i];
				bestMethod = classifiers[i];
			}
		}
		System.out.println("Best Algorithm: " + bestMethod + " with precision: " + max);
        
		pw.write(sb.toString());
        pw.close();
	}
	
	public static boolean validArff(String file) {
		//string must contain and end in .arff to be a valid arff file
		return (file.contains(".arff") && file.indexOf(".arff") == file.length() - 5);
	}
	
	public static boolean validCsv(String file) {
		return (file.contains(".csv") && file.indexOf(".csv") == file.length() - 4);
	}
	
	public static boolean fileExists(String file) {
		File newFile = new File(file);
		return newFile.exists();
	}
	
	public static String setFileName(String fileType, String outputDir) {
		String outputFile = "";
		boolean isValid = false;
		while(!isValid) {
			outputFile = outputDir + sc.next();	
			if(fileType.equals(".arff")) {
				if(validArff(outputFile) == isValid) {
					System.out.println("Sorry, the file you entered does not end in '.arff'. Please try again.");
					continue;
				}
			}
			if(fileType.equals(".csv")) {
				if(validCsv(outputFile) == isValid) {
					System.out.println("Sorry, the file you entered does not end in '.csv'. Please try again.");
					continue;
				}
			}
			if(fileExists(outputFile)) {
				System.out.println("This file already exists in this directory. Do you want to overwrite it? (Y/n)?");
				String answer = sc.next();
				if(answer.equals("Y") || answer.equals("y")) {
					isValid = true;
				}
				else if(answer.equals("N") || answer.equals("n")){
					System.out.println("Please enter another name.");
				}
				else {
					System.out.println("Could not understand input. Please enter a name again.");
				}
			}
			else {
				isValid = true;
			}
		}
		return outputFile;
	}
	
	public static Instances fileReader(String input) throws IOException {

		File inputFile = new File(input);
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances data = atf.getDataSet();
		
		return data;
		
	}

	public static String removeAttribute(Instances data, String [] attributes) throws Exception{
		
		String [] options = new String[2];
		options[0] = "-R";
		
		String indices = "";
		
		for(int i = 0; i < attributes.length; i++) {
			int index = (data.attribute(attributes[i])).index() + 1;
			indices += index;
			if(i != attributes.length-1) {
				indices += ",";
			}
		}
		
		return indices;
	}
	
	public static Double [] predict(Instances train, Instances test, String [] classifierNames, 
		String indicesToRemove, String classAttribute, String outputFile) throws Exception {
		
		//set the Class (what we want to predict)
		test.setClass(test.attribute(classAttribute));
		
		//setting the train class index to be the same as the testing class index
		train.setClassIndex(test.classIndex());
		
		//going to make an array of classifiers
		Classifier [] classifiers = new Classifier [classifierNames.length];
		
		//need somewhere to store the precision of each classifier + initializing the array
		Double precision [] = new Double [classifiers.length];
		for(int i = 0; i < precision.length; i++) {
			precision[i] = 0.0;
		}
		
		//removing attributes we don't want to include such as das and barcode 
		Remove rm = new Remove();
		rm.setAttributeIndices(indicesToRemove);
		
		//creating filtered versions of each classifier (removing das and barcode)
		for(int i = 0; i < classifiers.length; i++) {
			Classifier temp = AbstractClassifier.forName(classifierNames[i], null);
			
			FilteredClassifier fc = new FilteredClassifier();
			fc.setFilter(rm);
			fc.setClassifier(temp);
			
			classifiers[i] = fc;
		}
		
		//building the models for each classifier
		for(int i = 0; i < classifiers.length; i++) {
			classifiers[i].buildClassifier(train);
		}
		
		//writing the header to the output csv
		File output = new File(outputFile);
		PrintWriter pw = new PrintWriter(output);
        StringBuilder sb = new StringBuilder();
        sb.append("Barcode");
        sb.append(',');
        sb.append("Das");
        sb.append(",");
        sb.append("Actual");
        sb.append(',');
        
        //making the names of the classifiers part of the header, will indicate the stage that classifier has predicted for a particular plant
        for(int i = 0; i < classifierNames.length; i++) {
        	sb.append(classifierNames[i]);
        	if(i != classifierNames.length - 1) {
        		sb.append(",");
        	}
        }
        sb.append("\n");
		
        double numInst = test.numInstances();
        
        //actually running the classifier
    	for(int i = 0; i < numInst; i++){
    		
    		Instance current = test.instance(i);
			
    		//will set the Stage of this 'temp' instance to be the predicted value to then compare it to the actual value of 'current'
			Instance temp = (Instance)current.copy();
			
			//attributes are given as array positions, getting the string value
			String actualVal = current.stringValue(test.classIndex());
    		
    		sb.append((int) current.value(test.attribute("barcode")));
			sb.append(',');
			sb.append((int) current.value(test.attribute("das")));
			sb.append(",");
			sb.append(actualVal);
			sb.append(',');
    		
	       for(int j = 0; j < classifiers.length; j++) {
			//getting the predicted value of the class attribute of this instance
			double predicted = classifiers[j].classifyInstance(test.instance(i));
			
			//setting this value to the temp class attribute
			temp.setValue(test.classIndex(), predicted);
			
			//getting the string value
			String predictedVal = temp.stringValue(temp.classIndex());
			
			//comparing predicted with actual value
			if(predictedVal.equals(actualVal)) {
				precision[j]++;
			}
			
			sb.append(predictedVal);
			
			if(j != classifiers.length - 1) {
				sb.append(",");
			}
			
			else if( j == classifiers.length - 1) {
				sb.append('\n');
			}
				
		   }
    	}
    	
    	for(int i = 0; i < precision.length; i++) {
    		precision[i] = 100*precision[i]/numInst;
    	}
		
		sb.append('\n');
		pw.write(sb.toString());
        pw.close();
        
        return precision;
	}
		
	private static void connectToDatabase(String usrDB, String passwordDB, String conDB, String trainName, String testName) throws SQLException, FileNotFoundException {
		
		File trainingOutput = new File(trainName);
		PrintWriter trainingPw = new PrintWriter(trainingOutput);
        StringBuilder sb = new StringBuilder();
        
        File testingOutput = new File(testName);
        PrintWriter testingPw = new PrintWriter(testingOutput);
        
        sb.append("@relation databasetraining" + "\n" +  "\n" + "@attribute barcode numeric"  + "\n" + "@attribute area numeric" + "\n" + "@attribute perimeter numeric" + "\n" + 
        "@attribute circularity numeric" + "\n" + "@attribute compactness numeric" + "\n" + "@attribute major numeric" + "\n" + 
        "@attribute minor numeric" + "\n" + "@attribute eccentricity numeric" + "\n" + "@attribute hisgreypeak numeric" + "\n" + 
        "@attribute q1grey numeric" + "\n" + "@attribute q2grey numeric" + "\n" + "@attribute q3grey numeric" + "\n" + 
        "@attribute q1r numeric" + "\n" + "@attribute q2r numeric" + "\n" + "@attribute q3r numeric" + "\n" + "@attribute q1g numeric" + "\n" + 
        "@attribute q2g numeric" + "\n" + "@attribute q3g numeric" + "\n" + "@attribute q1b numeric" + "\n" + "@attribute q2b numeric" + "\n" + 
        "@attribute q3b numeric" + "\n" + "@attribute das numeric" + "\n" +"@attribute Stage {'Stage 1','Stage 2','Stage 3','Stage 4', 'Stage 5'}" + "\n" + "\n" + "@data" + "\n");
        
        trainingPw.write(sb.toString());
        testingPw.write(sb.toString());
        
	    try {
	    	Class.forName("org.postgresql.Driver");
	    	Connection conn = DriverManager.getConnection(conDB, usrDB, passwordDB);
		
	    	String trainingSql = "SELECT s.barcode, o.area, "
	    			+ "o.perimeter, o.circularity, o.compactness, "
	    			+ "o.major, o.minor, o.eccentricity, o.hisgreypeak, "
	    			+ "o.q1grey, o.q2grey, o.q3grey, "
	    			+ "o.q1r, o.q2r, o.q3r, "
	    			+ "o.q1g, o.q2g, o.q3g, "
	    			+ "o.q1b, o.q2b, o.q3b, d.das, "
	    			+ "CASE WHEN ( d.das <= 17 ) THEN 'Stage 1' "
	    			+ "WHEN ( d.das > 18 AND d.das <= 25 ) THEN 'Stage 2' "
	    			+ "WHEN ( d.das > 25 AND d.das <= 32 ) THEN 'Stage 3' "
	    			+ "WHEN ( d.das > 32 AND d.das <= 39 ) THEN 'Stage 4' "
	    			+ "WHEN ( d.das > 39 AND d.das <= 46 ) THEN 'Stage 5' "
	    			+ "ELSE 'Stage 6' END Stage "
	    			+ "FROM imageev AS i, imgobjectev AS o, soyidentification AS s, dasplusev AS d "
	    			+ "WHERE i.assayid = o.assayid "
	    			+ "AND i.imgid = o.imgid "
	    			+ "AND s.barcode = ( CAST( i.barcode AS INTEGER ) ) "
	    			+ "AND i.assayid = d.assayid AND i.fdate = d.fdate "
	    			+ "AND i.set = d.set "
	    			+ "AND s.line = 1 "
	    			+ "AND i.camera = 'vis-side-1-0' "
	    			+ "AND i.set = '3'";
	    	
	    	String testingSql =  "SELECT s.barcode, o.area, "
	    			+ "o.perimeter, o.circularity, o.compactness, "
	    			+ "o.major, o.minor, o.eccentricity, o.hisgreypeak, "
	    			+ "o.q1grey, o.q2grey, o.q3grey, "
	    			+ "o.q1r, o.q2r, o.q3r, "
	    			+ "o.q1g, o.q2g, o.q3g, "
	    			+ "o.q1b, o.q2b, o.q3b, d.das, "
	    			+ "CASE WHEN ( d.das <= 17 ) THEN 'Stage 1' "
	    			+ "WHEN ( d.das > 18 AND d.das <= 25 ) THEN 'Stage 2' "
	    			+ "WHEN ( d.das > 25 AND d.das <= 32 ) THEN 'Stage 3' "
	    			+ "WHEN ( d.das > 32 AND d.das <= 39 ) THEN 'Stage 4' "
	    			+ "WHEN ( d.das > 39 AND d.das <= 46 ) THEN 'Stage 5' "
	    			+ "ELSE 'Stage 6' END Stage "
	    			+ "FROM imageev AS i, imgobjectev AS o, soyidentification AS s, dasplusev AS d "
	    			+ "WHERE i.assayid = o.assayid "
	    			+ "AND i.imgid = o.imgid "
	    			+ "AND s.barcode = ( CAST( i.barcode AS INTEGER ) ) "
	    			+ "AND i.assayid = d.assayid "
	    			+ "AND i.fdate = d.fdate "
	    			+ "AND i.set = d.set "
	    			+ "AND ( s.line = 1 OR s.line = 2 OR s.line = 3 ) "
	    			+ "AND i.camera = 'vis-side-1-0' "
	    			+ "AND i.set = '2' "
	    			+ "AND d.das < 40 UNION "
	    			+ "SELECT s.barcode, o.area, o.perimeter, o.circularity, "
	    			+ "o.compactness, o.major, o.minor, o.eccentricity, o.hisgreypeak, "
	    			+ "o.q1grey, o.q2grey, o.q3grey, "
	    			+ "o.q1r, o.q2r, o.q3r,"
	    			+ " o.q1g, o.q2g, o.q3g, "
	    			+ "o.q1b, o.q2b, o.q3b, d.das, "
	    			+ "CASE WHEN ( d.das <= 17 ) THEN 'Stage 1' "
	    			+ "WHEN ( d.das > 18 AND d.das <= 25 ) THEN 'Stage 2' "
	    			+ "WHEN ( d.das > 25 AND d.das <= 32 ) THEN 'Stage 3' "
	    			+ "WHEN ( d.das > 32 AND d.das <= 39 ) THEN 'Stage 4' "
	    			+ "WHEN ( d.das > 39 AND d.das <= 46 ) THEN 'Stage 5' "
	    			+ "ELSE 'Stage 6' END Stage "
	    			+ "FROM imageev AS i, imgobjectev AS o, soyidentification AS s, dasplusev AS d "
	    			+ "WHERE i.assayid = o.assayid "
	    			+ "AND i.imgid = o.imgid "
	    			+ "AND s.barcode = ( CAST( i.barcode AS INTEGER ) ) "
	    			+ "AND i.assayid = d.assayid "
	    			+ "AND i.fdate = d.fdate "
	    			+ "AND i.set = d.set "
	    			+ "AND (s.line = 2 OR s.line = 3 ) "
	    			+ "AND i.camera = 'vis-side-1-0' "
	    			+ "AND i.set = '3' "
	    			+ "AND d.das < 40";
	    	
	    	PreparedStatement trainingPs = conn.prepareStatement(trainingSql);
			ResultSet trainingSet = trainingPs.executeQuery();
			while(trainingSet.next()) {
				Double barcode = trainingSet.getDouble("barcode");
				Double area = trainingSet.getDouble("area");
				Double perimeter = trainingSet.getDouble("perimeter");
				Double circularity = trainingSet.getDouble("circularity");
				Double compactness = trainingSet.getDouble("compactness");
				Double major = trainingSet.getDouble("major");
				Double minor = trainingSet.getDouble("minor");
				Double eccentricity = trainingSet.getDouble("eccentricity");
				Double hisgreypeak = trainingSet.getDouble("hisgreypeak");
				Double q1grey = trainingSet.getDouble("q1grey");
				Double q2grey = trainingSet.getDouble("q2grey");
				Double q3grey = trainingSet.getDouble("q3grey");
				Double q1r = trainingSet.getDouble("q1r");
				Double q2r = trainingSet.getDouble("q2r");
				Double q3r = trainingSet.getDouble("q3r");
				Double q1g = trainingSet.getDouble("q1g");
				Double q2g = trainingSet.getDouble("q2g");
				Double q3g = trainingSet.getDouble("q3g");
				Double q1b = trainingSet.getDouble("q1b");
				Double q2b = trainingSet.getDouble("q2b");
				Double q3b = trainingSet.getDouble("q3b");
				Double das = trainingSet.getDouble("das");
				String stage = trainingSet.getString("Stage");
				
				trainingPw.write(barcode + "," + area + ", " + perimeter + ", " + circularity + ", " + compactness + ", " + major + ", " + minor + ", " + eccentricity
						+ ", " + hisgreypeak + ", " + q1grey + ", " + q2grey + ", " + q3grey + ", " + q1r + ", " + q2r + ", " + q3r
						+ ", " + q1g + ", " + q2g + ", " + q3g + ", " + q1b + ", " + q2b + ", " + q3b + "," + das + "," + "'" + stage + "'" + "\n");
			}
			trainingSet.close();
			trainingPw.close();
			
			PreparedStatement  testingPs = conn.prepareStatement(testingSql);
			ResultSet testingSet = testingPs.executeQuery();
			while(testingSet.next()) {
				Double barcode = testingSet.getDouble("barcode");
				Double area = testingSet.getDouble("area");
				Double perimeter = testingSet.getDouble("perimeter");
				Double circularity = testingSet.getDouble("circularity");
				Double compactness = testingSet.getDouble("compactness");
				Double major = testingSet.getDouble("major");
				Double minor = testingSet.getDouble("minor");
				Double eccentricity = testingSet.getDouble("eccentricity");
				Double hisgreypeak = testingSet.getDouble("hisgreypeak");
				Double q1grey = testingSet.getDouble("q1grey");
				Double q2grey = testingSet.getDouble("q2grey");
				Double q3grey = testingSet.getDouble("q3grey");
				Double q1r = testingSet.getDouble("q1r");
				Double q2r = testingSet.getDouble("q2r");
				Double q3r = testingSet.getDouble("q3r");
				Double q1g = testingSet.getDouble("q1g");
				Double q2g = testingSet.getDouble("q2g");
				Double q3g = testingSet.getDouble("q3g");
				Double q1b = testingSet.getDouble("q1b");
				Double q2b = testingSet.getDouble("q2b");
				Double q3b = testingSet.getDouble("q3b");
				Double das = testingSet.getDouble("das");
				String stage = testingSet.getString("Stage");
				
				testingPw.write(barcode + "," + area + ", " + perimeter + ", " + circularity + ", " + compactness + ", " + major + ", " + minor + ", " + eccentricity
						+ ", " + hisgreypeak + ", " + q1grey + ", " + q2grey + ", " + q3grey + ", " + q1r + ", " + q2r + ", " + q3r
						+ ", " + q1g + ", " + q2g + ", " + q3g + ", " + q1b + ", " + q2b + ", " + q3b + "," + das + "," + "'" + stage + "'" + "\n");
			}
			testingSet.close();
			testingPw.close();
			
			conn.close();
		} 
	    catch (ClassNotFoundException e) {

			System.out.println("Improper database connection set-up.");
			e.printStackTrace();

		}
	}
}
