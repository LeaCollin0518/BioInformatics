/*
 * @author Lea Collin
 */
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.attributeSelection.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.HashMap;
import java.sql.*;

public class Driver {
	
	public static void main(String [] args) throws Exception {
		
		String dbUsr = args[1];
		String dbPwd = args[2];
		String dbConfig = args[3];
		
		//make these strings be taken in as program arguments
		
		String outputDir = args[0];
		
		String trainingFile = outputDir + "DatabaseTraining.arff";
		String testingFile = outputDir + "DatabaseTesting.arff";
		
		connectToDatabase(dbUsr, dbPwd, dbConfig, trainingFile, testingFile);
		
		//reading the files and getting all the instances of each one
		Instances train = fileReader(trainingFile);
		Instances test = fileReader(testingFile);
		
		String [] attributesToRemove = {"barcode", "das"};
		
		Instances instancesTrain = removeAttribute(train,attributesToRemove);
		Instances instancesTest = removeAttribute(test, attributesToRemove);
		
		String reduce = "no";
		
		if(reduce.equals("yes")){
			//selecting most relevant attributes
			//String evaluator = "cfs";
			//String evaluator = "corr";
			//String evaluator = "oner";
			//String evaluator = "principal";
			String evaluator = "relief";
			
			Instances [] reduced = attributeSelector(instancesTrain, instancesTest, evaluator);
			
			instancesTrain = reduced[0];
			
			instancesTest = reduced[1];
		}
		
		//what attribute do we want to predict
		String classAttribute = "Stage";
		
		String [] classifiers = {"ZeroR", "J48", "RandomTree", "RandomForest", "NaiveBayes"};
		
		//need to keep track of the precision of different algorithms
		HashMap <String, Double> precisionVals = new HashMap <String, Double>();
		double maxPrecision = 0.0;
		String bestMethod = "";
		
		//running each different classifier, population HashMap to store each precision value
		for(int i = 0; i < classifiers.length; i++) {
			String toTest = classifiers[i];
			precisionVals.put(toTest, predict(instancesTrain, instancesTest, toTest, null, classAttribute, outputDir));
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
	
	public static Instances removeAttribute(Instances data, String [] attributes) throws Exception{
		
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
		
		options[1] = indices;
		
		Remove remove = new Remove();
		remove.setOptions(options);
		remove.setInputFormat(data);
		Instances newData = Filter.useFilter(data, remove);
		
		return newData;
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
		
		for(int i = 0; i < numInst; i++){
			
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
	
	public static Instances [] attributeSelector(Instances train, Instances test, String evaluator) throws Exception {
		
		AttributeSelection selector = new AttributeSelection();
		
		if (evaluator.equals("cfs")) {
			CfsSubsetEval eval = new CfsSubsetEval();
			BestFirst search = new BestFirst();
			selector.setEvaluator(eval);
			selector.setSearch(search);
		}
		else if(evaluator.equals("corr")) {
			CorrelationAttributeEval eval = new CorrelationAttributeEval();
	        Ranker search = new Ranker();
	        selector.setEvaluator(eval);
			selector.setSearch(search);
		}
		else if(evaluator.equals("oner")) {
			OneRAttributeEval eval = new OneRAttributeEval();
	        Ranker search = new Ranker();
	        selector.setEvaluator(eval);
			selector.setSearch(search);
		}
		/*need to fix this
		 * else if(evaluator.equals("principal")) {
			PrincipalComponents eval = new PrincipalComponents();
	        Ranker search = new Ranker();
	        selector.setEvaluator(eval);
			selector.setSearch(search);
		}*/
		else if(evaluator.equals("relief")) {
			ReliefFAttributeEval eval = new ReliefFAttributeEval();
	        Ranker search = new Ranker();
	        selector.setEvaluator(eval);
			selector.setSearch(search);
		}
		
        
		selector.SelectAttributes(train);
		
		//rankedAttributes gives the ranking of the attributes along with their weights
		//selected Attributes just gives the order of the ranking

		
		Instances trainTemp = selector.reduceDimensionality(train);
		Instances trainTest = selector.reduceDimensionality(test);
		
		
		Instances [] reduced = {trainTemp, trainTest};

		return reduced;
	}
	
	private static void connectToDatabase(String usrDB, String passwordDB, String conDB, String trainingFile, String testingFile) throws SQLException, FileNotFoundException {
		
		File trainingOutput = new File(trainingFile);
		PrintWriter trainingPw = new PrintWriter(trainingOutput);
        StringBuilder sb = new StringBuilder();
        
        File testingOutput = new File(testingFile);
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
				Integer barcode = trainingSet.getInt("barcode");
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
				Integer barcode = testingSet.getInt("barcode");
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
			return;

		}
	}	
}
