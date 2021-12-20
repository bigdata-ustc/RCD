I. Authorization:
	Any form of commercial usage is not allowed!
	Please cite the following paper if you publish your work:

	Haw-Shiuan Chang, Hwai-Jung Hsu and Kuan-Ta Chen,
	"Modeling Exercise Relationships in E-Learning: A Unified Approach,"
	International Conference on Educational Data Mining (EDM), 2015.
-----------------------------------------------------
II. Introduction:
	The dataset contains the problem log and exercise-related information on the Junyi Academy ( http://www.junyiacademy.org/ ), an E-learning platform established in 2012 on the basis of the open-source code released by Khan Academy. In addition, the annotations of exercise relationship we collected for building models are also available. 
-----------------------------------------------------
III. Meaning of Fields
	junyi_Exercise_table.csv:
		name:	Exercise name (The name is also an id of exercise, so each name is unique in the dataset). If you want to access the exercise on the website, please append this name after url, http://www.junyiacademy.org/exercise/ (e.g., http://www.junyiacademy.org/exercise/similar_triangles_1 ). Please note that Junyi Academy are constantly changing their contents as Khan Academy did, so some url of exercises might be unavaible when you access them.
		live: Whether the exercise is still accessible on the website on Jan. 2015
		prerequisite:	Indicate its prerequisite exericse (parent shown in its knowledge map)
		h_position:	The coordiate on the x axis of the knowledge map
		v_position:	The coordiate on the y axis of the knowledge map
		creation_date:	The date this exercise is created
		seconds_per_fast_problem:	The website judge a student finish the exercise fast if he/she takes less then this time to answer the question. The number is manually assigned by the experts in Junyi Academy.
		pretty_display_name:	The chinese name of exercise shown in the knowledge map (Please use UTF-8 to decode the chinese characters)
		short_display_name:	Another chinese name of exercise (Please use UTF-8 to decode the chinese characters)
		topic:	The topic of each exercise, and the topic would be shown as a larger node in the knowledge map.
		area:	The area of each exercise (Each area contains several topics)
	relationship_annotation_training.csv / relationship_annotation_testing.csv
		Exercise_A, Exercise_B:	The exercise names being compared
		Similarity_avg, Difficulty_avg, Prequesite_avg:	The mean opinion scores of different relationships. This is also the ground truth we used to train/test our model.
		Similarity_raw, Difficulty_raw, Prequesite_raw:	The raw scores given by workers (delimiter is "_")
	junyi_ProblemLog_original.zip
		user_id:	An number represents an user
		exercise:	Exercise name
		problem_type:	Some exercises would record what template of problem this student encounters at this time
		problem_number:	How many times this student practices this exercise (e.g., the number would be 1 if the student tries to answer this exercise at the first time)
		topic_mode: Whether the student is assigned this exercise by clicking the topic icon (This function has been closed now)
		suggested:	Whether the exercise is suggested by the system according to prerequisite relationships on the knowledge map
		review_mode:	Whether the exercise is done by the student after he/she earn proficiency
		time_done:		Unix timestamp in microsecends
		time_taken:		Second the student spend on this exercise
		time_taken_attempts:	Seconds the student spend on each answering attempt 
		correct:	Whether the student's first attempt is correct, and the field would be false if any hint is requested
		count_attempts:	How many times student attempt to answer the problem
		hint_used:	Whether student request hints
		count_hints:	How many times student request hints
		hint_time_taken_list:	Seconds the student spend on each requested hints
		earned_proficiency:	Whether the student reaches proficiency. Please refer to http://david-hu.com/2011/11/02/how-khan-academy-is-using-machine-learning-to-assess-student-mastery.html for the algorithm of determining proficiency 
		points_earned:	How many points students earn for this practice
	junyi_ProblemLog_for_PSLC.zip
		The tab delimited format used in PSLC datashop, please refer to their document ( https://pslcdatashop.web.cmu.edu/help?page=importFormatTd )
		The size of the text file is too large (9.1 GB) to analyze using tools of websites, so we compress the text file and put it as an extra file of the dataset. We also upload a small subset of data into the website for the illustration purpose. Note that there are some assumptions when converting the data into this format, please read the description of our dataset for more details.

-----------------------------------------------------
IV. Questions and Collaboration:
	1. If you have any question to this dataset, please e-mail to hschang@cs.umass.edu.
	2. If you have intention to acquire more data which fit your research purpose, please contact Junyi Academy directly for discussing the further cooperation opportunites by emailing to support@junyiacademy.org
-----------------------------------------------------
V Note:
	1. The dataset we used in our paper (Modeling Exercise Relationships in E-Learning: A Unified Approach) is extracted from Junyi Academy on July 2014, and this dataset is extracted on Jan 2015. After applying our method on the new dataset, we got similar observation with that in our paper, even though this dataset contains more users and exercises. 
	2. After uncompress the original problem log and problem log using PLSC format, the text files will take around 2.6 GB and 9.1 GB respectively. Please prepare enough space in your disk.