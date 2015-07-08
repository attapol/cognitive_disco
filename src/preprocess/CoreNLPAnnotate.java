package preprocess;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.io.FileUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.TypesafeMap.Key;


public class CoreNLPAnnotate {

	StanfordCoreNLP pipeline;
	public CoreNLPAnnotate() {
		// TODO Auto-generated constructor stub
		Properties props = new Properties();
		//props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref, sentiment");
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
		props.setProperty("tokenize.whitespace", "true");
		props.setProperty("ssplit.eolonly", "true");
		pipeline = new StanfordCoreNLP(props);
	}
	
	private String getTokens(JSONArray sentencesJSON) throws JSONException{
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < sentencesJSON.length(); i++){
			JSONObject sentence = (JSONObject) sentencesJSON.get(i);
			JSONArray wordsJSON = sentence.getJSONArray("words");
			for (int j = 0; j < wordsJSON.length(); j++){
				String word = wordsJSON.getJSONArray(j).getString(0);
				sb.append(word + " ");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
	
	public void annotate(String fileName, String outputFileName) throws IOException, JSONException {
		String jsonString = FileUtils.readFileToString(new File(fileName));
		JSONObject parseData = new JSONObject(jsonString);	
		Iterator kit = parseData.keys();
		while (kit.hasNext()){
			String docID = (String) kit.next();
			JSONArray sentencesJSON = parseData.getJSONObject(docID).getJSONArray("sentences");

			Annotation annotation = new Annotation(getTokens(sentencesJSON));
			pipeline.annotate(annotation);
			List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
			
			/*
			Map<Integer, CorefChain> corefChain = annotation.get(CorefChainAnnotation.class);
			System.out.println(corefChain);
			*/

			for(int i = 0; i < sentences.size(); i++) {
				CoreMap sentence = sentences.get(i);

				/*
				Tree stree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
				System.out.println(stree.toString());
				*/

				// traversing the words in the current sentence
				// a CoreLabel is a CoreMap with additional token-specific methods
				JSONObject sentenceJSON = (JSONObject) sentencesJSON.get(i);
				List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
				for (int j = 0; j < tokens.size(); j++) {
					CoreLabel token = tokens.get(j);
					String word = token.get(TextAnnotation.class);
					String pos = token.get(PartOfSpeechAnnotation.class);
					String ne = token.get(NamedEntityTagAnnotation.class);       
					String lemma = token.get(LemmaAnnotation.class);

					//System.out.println(word +" " + lemma+ " " + pos + " " + ne );
					
					JSONArray wordJSON = sentenceJSON.getJSONArray("words").getJSONArray(j);
					JSONObject wordInfo = wordJSON.getJSONObject(1);

					wordInfo.put("NE", ne);
					wordInfo.put("Lemma", lemma);
				} // loop over words

				// this is the parse tree of the current sentence
				//Tree tree = sentence.get(TreeAnnotation.class);
				// this is the Stanford dependency graph of the current sentence
				//SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);

			} // loop over sentences
		} // loop over documents
		FileWriter wr = new FileWriter(new File(outputFileName));
		parseData.write(wr);
		wr.close();
		//System.out.println(parseData.toString(2));
			
	}
	
	public static void main(String args[]) throws IOException, JSONException {
		CoreNLPAnnotate cnlp = new CoreNLPAnnotate();
		cnlp.annotate(args[0], args[1]);
		/*
		for (String fileName : args){
			cnlp.annotate(fileName);
		}
		*/
		
	}

}
