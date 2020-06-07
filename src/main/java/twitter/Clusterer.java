package twitter;


import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Clusterer {
	
	// Configure spark to run locally
	static SparkConf sparkConf = new SparkConf().setAppName("TwitterCluster").setMaster("local[4]").set("spark.executor.memory", "1g");
	static JavaSparkContext sc = new JavaSparkContext(sparkConf);
	
	static final String path = "src/twitter2D.txt";
	
	public static void kCluster() {
		// Only log errors so console is not plagued by red lines
		sc.setLogLevel("ERROR");
		
		// Read data from text file
		JavaRDD<String> data = sc.textFile(path, 1);

		// Stream through each line of the data from the text file, split it by the comma,
		// and put the first and second values as the coordinates into a vector for each line
		JavaRDD<Vector> parsedData = data.map((String s) -> {
			String[] sarray = s.split(",");
			double[] values = {Double.parseDouble(sarray[0]), Double.parseDouble(sarray[1])};
			return Vectors.dense(values);
		});
		
		// Cache the data
		parsedData.cache();
		
		// Set the number of clusters and iterations, which are used as kmeans clustering parameters
		int numClusters = 4;
		int numIterations = 20;	
		
		// Create and train a model using the parsed data
		KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);	
			
		// Like before, create a vector from the coordinate values, but this time feed them values to our clustering model which returns the cluster number
		// a set of coordinates belongs to. Add the text from the tweet to a tuple as it's key, and it's cluster number as it's value.
		JavaRDD<Tuple2<String, Integer>> predictions = data.map((String s) -> {
			String[] sarray = s.split(",");
			double[] values = {Double.parseDouble(sarray[0]), Double.parseDouble(sarray[1])};
			Vector v = Vectors.dense(values);
			int i = clusters.predict(v);
			return new Tuple2<String, Integer>(sarray[sarray.length-1], i);
		// Use RDD sortBy method which takes in a Function object specifying on the order of the sort
		}).sortBy(new Function<Tuple2<String,Integer>,Integer>() {			
			private static final long serialVersionUID = 1L;
			public Integer call(Tuple2<String,Integer> t) throws Exception {
				return t._2;
			}
		}, true, 1);;
		
		// Print each tweet and its' cluster to the console
		predictions.foreach((Tuple2<String,Integer> t) -> {
			System.out.println("Tweet \"" + t._1 + "\" is in cluster " + t._2);
		});	
	}
	
	public static void main(String[] args) {
		Clusterer.kCluster();
	}
}
