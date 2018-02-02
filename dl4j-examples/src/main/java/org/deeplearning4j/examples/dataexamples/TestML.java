package org.deeplearning4j.examples.dataexamples;

import java.io.IOException;
import java.util.Arrays;

import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

/**
 * Read a csv file. Fit and plot the data using Deeplearning4J.
 *
 * @author Matt Sordello, Scott Surette
 */
public class TestML {
	/**
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {
			SparkSession spark = SparkSession.builder()
				.appName("SparkSample")
				.master("local[*]")
				.getOrCreate();

		SQLContext sqlContext = new SQLContext(spark);

		Dataset<Row> ds = sqlContext.read()
				.format("csv")
				.option("inferSchema", "true")
				.option("header", "true")
				.load("src/main/resources/DataExamples/Iowa_Property_Casualty_Insurance_Premiums_and_Losses.csv");

		StringIndexerModel indexer = new StringIndexer()
				.setInputCol("State")
				.setOutputCol("StateIndex")
				.fit(ds);
		Dataset<Row> indexed = indexer.transform(ds);

		OneHotEncoder encoder = new OneHotEncoder()
				.setInputCol("StateIndex")
				.setOutputCol("StateIndexVec");
		Dataset<Row> encoded = encoder.transform(indexed);

		StringIndexerModel indexerCN = new StringIndexer()
				.setInputCol("Company Name")
				.setOutputCol("CompanyNameIndex")
				.fit(encoded);
		Dataset<Row> indexedCN = indexerCN.transform(encoded);

		OneHotEncoder encoderCN = new OneHotEncoder()
				.setInputCol("CompanyNameIndex")
				.setOutputCol("CompanyNameIndexVec");
		Dataset<Row> encodedCN = encoderCN.transform(indexedCN);

		StringIndexerModel indexerLOB = new StringIndexer()
				.setInputCol("Line of Insurance")
				.setOutputCol("LOBIndex")
				.fit(encoded);
		Dataset<Row> indexedLOB = indexerLOB.transform(encodedCN);

		OneHotEncoder encoderLOB = new OneHotEncoder()
				.setInputCol("LOBIndex")
				.setOutputCol("LOBIndexVec");
		Dataset<Row> encodedLOB = encoderLOB.transform(indexedLOB);

		encodedLOB.show();
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] { "StateIndexVec", "LOBIndexVec", 
						"Iowa Company Code", "NAIC Number", "Taxes Paid", "Premiums Written" }).setOutputCol("features");
		Dataset<Row> output = assembler.transform(encodedLOB);
		// Set parameters for the algorithm.
		// Here, we limit the number of iterations to 10.

		GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
				.setFamily("gaussian")
				.setLink("identity")
				.setMaxIter(50)
				.setRegParam(1)
				.setLabelCol("Losses Paid");

		// Fit the model
		GeneralizedLinearRegressionModel model = glr.fit(output);

		GeneralizedLinearRegressionTrainingSummary summary = model.summary();

		model.transform(output).show();

		System.out.println("Coefficient Standard Errors: " + Arrays.toString(summary.coefficientStandardErrors()));
		System.out.println("T Values: " + Arrays.toString(summary.tValues()));
		System.out.println("P Values: " + Arrays.toString(summary.pValues()));
		System.out.println("Dispersion: " + summary.dispersion());
		System.out.println("Null Deviance: " + summary.nullDeviance());
		System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
		System.out.println("Deviance: " + summary.deviance());
		System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
		System.out.println("AIC: " + summary.aic());
		System.out.println("Deviance Residuals: ");
		summary.residuals().show();

	}
}
