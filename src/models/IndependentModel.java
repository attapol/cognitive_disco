package models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.DataTriplet;
import types.LabelType;
import types.SimpleConfusionMatrix;
import cc.mallet.classify.evaluate.ConfusionMatrix;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.types.InstanceList;


/*
 * Independent model consists of a classifier for each dimension.
 * Each classifier uses the same set of features. 
 */
public abstract class IndependentModel extends CognitiveDiscourseParserBase{

	
	Classifier[] classifiers = null;

	public IndependentModel(String experimentName, String mappingJson,
			String trainingDir, String devDir, String testDir)
			throws JSONException, IOException {
		super(experimentName, mappingJson, trainingDir, devDir, testDir);
	}
	
	abstract public ClassifierTrainer<?> getNewTrainer();

	public ArrayList<ArrayList<String>> classifyDimensions(String dataFile, LabelType labelType) throws FileNotFoundException {
		if (classifiers == null){
			System.err.println("The classifiers have not been loaded or trained. Returning empty results");
			return null;
		}

		int numDimensions = classifiers.length;
		String[][] dimensionClassifications = new String[numDimensions][];
        int numInstances = 0;
		for (int i = 0; i < numDimensions; i++) {
			InstanceList unseenData = DataTriplet.importData(dataFile, labelType, classifiers[i].getAlphabet());

			numInstances = unseenData.size();
			dimensionClassifications[i] = new String[numInstances];
			for (int j = 0; j < numInstances; j++) {
				dimensionClassifications[i][j] = 
						(String) classifiers[i].classify(unseenData.get(j)).getLabeling().getBestLabel().getEntry();
			}
		}
		
		// Tranpose the matrix above and convert it to an ArrayList
		ArrayList<ArrayList<String>> predictedDimensions = new ArrayList<ArrayList<String>>(numInstances);
		for (int i = 0; i < numInstances; i++){
			ArrayList<String> dimensions = new ArrayList<String>();
			for (int d = 0; d < numDimensions; d++){
				dimensions.add(dimensionClassifications[d][i]);
			}
			predictedDimensions.add(dimensions);
		}
		return predictedDimensions;
	}

	public String[] classify(String dataFile, LabelType labelType) throws FileNotFoundException {
		ArrayList<ArrayList<String>> predictedDimensions = classifyDimensions(dataFile, labelType);
		int numInstances = predictedDimensions.size();
		String[] predictedLabels = new String[numInstances];
		for (int i = 0; i < numInstances; i++){
			predictedLabels[i] = dm.getLabel(predictedDimensions.get(i), labelType);
		}
		return predictedLabels;
	}

	
	public void trainTest() throws FileNotFoundException{
		// Train on each dimension
		int numDimensions = this.data.length;
		Trial[] trials = new Trial[numDimensions];
		classifiers = new Classifier[numDimensions];
		for (int i = 0; i < numDimensions ; i++) {
			ClassifierTrainer<?> trainer = getNewTrainer();
			this.data[i].importData();
			trainer.train(this.data[i].getTrainingSet());
			classifiers[i] = trainer.getClassifier();
			trials[i] = new Trial(classifiers[i], this.data[i].getDevSet());
		}
		
		// Evaluate on each dimension individually
		System.out.println("====== Individual classifier performance ======");
		for (Trial trial : trials){
			ConfusionMatrix cm = new ConfusionMatrix(trial);
			System.out.println(cm.toString());
		}
		
		String[] predictedLabels;
		String[] trueLabels;
		Trial baselineResult;
		for (LabelType labelType : LabelType.values()){
			// Evaluate on the combined dimension and mapped to the original label
			System.out.println("====== Dimension-based classifier performance ("+ labelType + ") ======");
			this.originalData.importData(labelType);

			trueLabels = DataTriplet.getStringLabels(this.originalData.getDevSet());
			predictedLabels = this.classify(this.originalData.getDevFileName(), labelType);
			SimpleConfusionMatrix cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
			System.out.println(cm.toString());

			System.out.println("====== Baseline classifier performance ("+ labelType +") ======");
			ClassifierTrainer<?> trainer = getNewTrainer();
			trainer.train(this.originalData.getTrainingSet());
			Classifier classifier = trainer.getClassifier();
			baselineResult = new Trial(classifier, this.originalData.getDevSet());
			predictedLabels = DataTriplet.getStringLabels(baselineResult);
			cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
			System.out.println(cm.toString());

		}
		
	}
}
