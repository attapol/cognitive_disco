package models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.DataTriplet;
import types.SimpleConfusionMatrix;
import cc.mallet.classify.evaluate.ConfusionMatrix;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.Csv2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.InstanceList;


public abstract class IndependentModel extends CognitiveDiscourseParserBase{

	
	Classifier[] classifiers = null;

	public IndependentModel(String experimentName, String mappingJson,
			String trainingDir, String devDir, String testDir)
			throws JSONException, IOException {
		super(experimentName, mappingJson, trainingDir, devDir, testDir);
	}
	
	abstract public ClassifierTrainer<?> getNewTrainer();

	@Override
	public String[] classify(String dataFile) throws FileNotFoundException {
		if (classifiers == null){
			System.err.println("The classifiers have not been loaded or trained. Returning empty results");
			return null;
		}

		int numDimensions = classifiers.length;
		String[][] dimensionClassifications = new String[numDimensions][];
        int numInstances = 0;
		for (int i = 0; i < numDimensions; i++) {
			Pipe pipe = new Csv2FeatureVector();
			pipe.setDataAlphabet(classifiers[i].getAlphabet());
			pipe.setTargetProcessing(false);
			InstanceList unseenData = DataTriplet.importData(dataFile, pipe);

			numInstances = unseenData.size();
			dimensionClassifications[i] = new String[numInstances];
			for (int j = 0; j < numInstances; j++) {
				dimensionClassifications[i][j] = 
						(String) classifiers[i].classify(unseenData.get(j)).getLabeling().getBestLabel().getEntry();
			}
		}
		
		String[] predictedLabels = new String[numInstances];
		for (int i = 0; i < numInstances; i++){
			ArrayList<String> dimensions = new ArrayList<String>();
			for (int d = 0; d < numDimensions; d++){
				dimensions.add(dimensionClassifications[d][i]);
			}
			predictedLabels[i] = dm.getTopLevelLabel(dimensions);
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
			trainer.train(this.data[i].getTrainingSet());
			classifiers[i] = trainer.getClassifier();
			trials[i] = new Trial(classifiers[i], this.data[i].getDevSet());
		}
		
		// Train on the original label
		ClassifierTrainer<?> trainer = getNewTrainer();
		trainer.train(this.originalData.getTrainingSet());
		Classifier classifier = trainer.getClassifier();
		Trial baselineResult = new Trial(classifier, this.originalData.getDevSet());
		
		// Evaluate on each dimension individually
		System.out.println("====== Individual classifier performance ======");
		for (Trial trial : trials){
			ConfusionMatrix cm = new ConfusionMatrix(trial);
			System.out.println(cm.toString());
		}
		
		// Evaluate on the combined dimension and mapped to the original label
		String[] predictedLabels = this.classify(this.originalData.getDevFileName());
		String[] trueLabels = DataTriplet.getStringLabels(this.originalData.getDevSet());
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
		System.out.println("====== Dimension-based classifier performance ======");
		System.out.println(cm.toString());
		
		System.out.println("====== Baseline classifier performance ======");

		predictedLabels = DataTriplet.getStringLabels(baselineResult);
		cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
		System.out.println(cm.toString());
		
	}

}