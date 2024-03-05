# from flask import Flask, render_template, request, jsonify


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_Chat_response(input)


# def get_Chat_response(text):

#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#         # generated a response while limiting the total chat history to 1000 tokens, 
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last ouput tokens from bot
#         return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


# if __name__ == '__main__':
#     app.run()






from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from openai import OpenAI  

app = Flask(__name__)


intents = {
    "who developed you":{
        "patterns":["who created you","who developed you","who invents you", "who crafted you","who makes you"],
        "responses":["I was developed by a team of AI Avengers.","I am an AI language model developed by a team of AI Avengers"]
    },
    "who are you":{
        "patterns":["who are you","hu are you","hu r you", "hu r u","hu are u"],
        "responses":["I an a MinesBot,","I am an AI language model named MinesBot."]
    },
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hey there", "what's up", "hello there"],
        "responses": ["Hello! How can I assist you?", "Hi there!", "Hey! What can I do for you?", "Howdy! What brings you here?", "Greetings! How may I help you?", "Good morning! How can I be of service?", "Good afternoon! What do you need assistance with?", "Good evening! How may I assist you?", "Hey there! How can I help?", "Hi! What's on your mind?", "Hello there! How can I assist you today?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye", "farewell", "take care", "until next time", "bye bye", "catch you later", "have a good one", "so long"],
        "responses": ["Goodbye!", "See you later!", "Have a great day!", "Farewell! Take care.", "Goodbye! Until next time.", "Take care! Have a wonderful day.", "Bye bye!", "Catch you later!", "Have a good one!", "So long!"]
    },
    "gratitude": { 
        "patterns": ["thank you", "thanks", "appreciate it", "thank you so much", "thanks a lot", "much appreciated"],
        "responses": ["You're welcome!", "Happy to help!", "Glad I could assist.", "Anytime!", "You're welcome! Have a great day.", "No problem!"]
    },
    "apologies": {
        "patterns": ["sorry", "my apologies", "apologize", "I'm sorry"],
        "responses": ["No problem at all.", "It's alright.", "No need to apologize.", "That's okay.", "Don't worry about it.", "Apology accepted."]
    },
    "mining_amendments": {
        "patterns": ["What amendments were introduced by G.S.R. 737(E)?",
                     "Can you explain the modifications made by G.S.R. 737(E)?",
                     "Tell me about the amendments in the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
                     "What changes were made to the mining regulations by G.S.R. 737(E)?",
                     "Explain the modifications introduced by G.S.R. 737(E) in the mining rules."],
        "responses": ["G.S.R. 737(E) introduced amendments to the Mines and Minerals (Development and Regulation) Act, 1957.",
                      "The Central Government, under section 13 of the Mines and Minerals (Development and Regulation) Act, 1957, made modifications to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
                      "Modifications were made to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023 by G.S.R. 737(E).",
                      "G.S.R. 737(E) brought changes to the mining regulations specified in the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
                      "The amendments introduced by G.S.R. 737(E) pertain to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023."]
    },
    "rule_44_modification": {
        "patterns": ["What modification was made to Rule 44 by G.S.R. 737(E)?",
                     "Can you explain clause (ia) of Rule 44 as per G.S.R. 737(E)?",
                     "Tell me about the changes introduced to Rule 44 by G.S.R. 737(E)."],
        "responses": ["G.S.R. 737(E) introduced clause (ia) to Rule 44 regarding Lithium.",
                      "Rule 44 was modified by G.S.R. 737(E) to include clause (ia) concerning Lithium.",
                      "The modification made to Rule 44 by G.S.R. 737(E) involves the addition of clause (ia) related to Lithium."]
    },
    "rule_45_modification": {
        "patterns": ["Explain the amendments to Rule 45 by G.S.R. 737(E).",
                     "What changes were made to sub-rules 5 and 6 of Rule 45 according to G.S.R. 737(E)?",
                     "Can you provide details about the modifications made to Rule 45 by G.S.R. 737(E)?"],
        "responses": ["G.S.R. 737(E) introduced modifications to Rule 45, including sub-rules 5 and 6.",
                      "Rule 45 was amended by G.S.R. 737(E) with changes to sub-rules 5 and 6.",
                      "The amendments made to Rule 45 by G.S.R. 737(E) include modifications to sub-rules 5 and 6."]
    },
    "publication_history": {
        "patterns": ["What is the publication history of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016?",
                     "Can you provide details about the publication of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016?",
                     "Tell me about the Gazette publication of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016."],
        "responses": ["The Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016 were published in the Gazette of India on March 4, 2016, under number G.S.R. 279(E)."]
    },
    "mining_amendments_736": {
        "patterns": ["What amendments were introduced by G.S.R. 736(E)?",
                     "Can you explain the modifications made by G.S.R. 736(E)?",
                     "Tell me about the amendments in the Mines and Minerals (Development and Regulation) Act, 1957 introduced by G.S.R. 736(E).",
                     "What changes were made to the mining regulations by G.S.R. 736(E)?",
                     "Explain the modifications introduced by G.S.R. 736(E) in the mining laws."],
        "responses": ["G.S.R. 736(E) introduced amendments to the Mines and Minerals (Development and Regulation) Act, 1957.",
                      "The Central Government, under sub-section (3) of section 9 of the Mines and Minerals (Development and Regulation) Act, 1957, made modifications to the mining regulations.",
                      "Modifications were made to the Mines and Minerals (Development and Regulation) Act, 1957 by G.S.R. 736(E).",
                      "G.S.R. 736(E) brought changes to the mining regulations specified in the Mines and Minerals (Development and Regulation) Act, 1957.",
                      "The amendments introduced by G.S.R. 736(E) pertain to the Mines and Minerals (Development and Regulation) Act, 1957."]
    },
    "item_28A_amendment": {
        "patterns": ["What amendment was made for Lithium by G.S.R. 736(E)?",
                     "Can you explain the modification introduced for Lithium by G.S.R. 736(E)?",
                     "Tell me about the change in charge for Lithium as per G.S.R. 736(E)."],
        "responses": ["G.S.R. 736(E) introduced a charge of three per cent of London Metal Exchange price on the Lithium metal in the ore produced."]
    },
    "item_33_amendment": {
        "patterns": ["What change was made to Monazite by G.S.R. 736(E)?",
                     "Can you explain the modification introduced for Monazite by G.S.R. 736(E)?",
                     "Tell me about the amendment regarding Monazite as per G.S.R. 736(E)."],
        "responses": ["G.S.R. 736(E) changed 'Monazite' to 'Monazite occurring in beach sand minerals'."]
    },
    "item_34A_amendment": {
        "patterns": ["Explain the amendments made for Niobium by G.S.R. 736(E).",
                     "What changes were introduced for Niobium by G.S.R. 736(E)?",
                     "Can you provide details about the modifications for Niobium by G.S.R. 736(E)?"],
        "responses": ["G.S.R. 736(E) introduced amendments for Niobium, specifying charges for both primary and by-product Niobium."]
    },
    "item_38A_amendment": {
        "patterns": ["What amendment was made for Rare Earth Elements by G.S.R. 736(E)?",
                     "Can you explain the modification introduced for Rare Earth Elements by G.S.R. 736(E)?",
                     "Tell me about the change in charge for Rare Earth Elements as per G.S.R. 736(E)."],
        "responses": ["G.S.R. 736(E) introduced a charge of one per cent of average sale price of Rare Earth Oxide on the Rare Earth Oxide contained in the ore produced."]
    },
    "offshore_mining_act_commencement": {
        "patterns": ["When does the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 come into force?",
                     "Can you tell me the commencement date of the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023?",
                     "When will the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 be effective?"],
        "responses": ["The Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 comes into force on the 17th of August, 2023."]
    },
    "mines_and_minerals_act_commencement": {
        "patterns": ["When does the Mines and Minerals (Development and Regulation) Amendment Act, 2023 come into force?",
                     "Can you tell me the commencement date of the Mines and Minerals (Development and Regulation) Amendment Act, 2023?",
                     "When will the Mines and Minerals (Development and Regulation) Amendment Act, 2023 be effective?"],
        "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023 comes into force on the 17th of August, 2023."]
    },
    "signatory": {
        "patterns": ["Who signed S.O. 3684(E)?",
                     "Can you provide the name of the signatory for S.O. 3684(E)?",
                     "Tell me who signed the document S.O. 3684(E)."],
        "responses": ["S.O. 3684(E) was signed by Dr. Veena Kumari Dermal, Joint Secretary."]
    },
    "act_details": {
        "patterns": ["Mines and Minerals (Development and Regulation) Amendment Act, 2023", "details about Mines and Minerals (Development and Regulation) Amendment Act, 2023", "Mines and Minerals Act 2023", "MMDR Amendment Act 2023"],
        "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023, received the assent of the President on August 9, 2023. It aims to further amend the Mines and Minerals (Development and Regulation) Act, 1957."]
    },
    "act_amendments": {
        "patterns": ["What are the amendments in the Mines and Minerals Act?", "Amendments in Mines and Minerals Act 2023", "Changes in Mines and Minerals Act", "MMDR Act amendments"],
        "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023, introduced several amendments to the Mines and Minerals (Development and Regulation) Act, 1957. These include amendments related to exploration licenses, termination of prospecting licenses and mining leases, restrictions on mineral concessions, maximum area for mineral concessions, procedure for obtaining mineral concession, application for mineral concession, grant of exploration licenses through auction, and more."]
    },
    "offshore_areas_mineral_act": {
        "patterns": ["What is the Offshore Areas Mineral (Development and Regulation) Act, 2002?", "Can you provide information about the Offshore Areas Mineral Act?", "What does the Offshore Areas Mineral Act govern?"],
        "responses": ["The Offshore Areas Mineral (Development and Regulation) Act, 2002, governs the development and regulation of mineral resources in India's offshore areas, including territorial waters, continental shelf, exclusive economic zone, and other maritime zones."]
    },
    "short_title_and_commencement": {
        "patterns": ["What is the short title of the Offshore Areas Mineral Act?", "When did the Offshore Areas Mineral Act come into force?"],
        "responses": ["The short title of the Offshore Areas Mineral Act is the Offshore Areas Mineral (Development and Regulation) Act, 2002. It came into force on the date specified by the Central Government through a notification in the Official Gazette."]
    },
    "expediency_of_union_control": {
        "patterns": ["Why was it declared expedient for the Union to control regulation of mines and mineral development in offshore areas?"],
        "responses": ["It was declared expedient in the public interest for the Union to control the regulation of mines and mineral development in offshore areas to the extent provided in the act."]
    },
    "application": {
        "patterns": ["To which areas does the Offshore Areas Mineral Act apply?", "What is the scope of the Offshore Areas Mineral Act?"],
        "responses": ["The Offshore Areas Mineral Act applies to all minerals in offshore areas, excluding mineral oils and hydrocarbons related thereto."]
    },
    "population": {
        "patterns": ["What is the population of the United States?", "Population of USA", "How many people live in the USA?", "What's the current population of the USA?"],
        "responses": ["The population of the United States is approximately 331 million people as of 2022."]
    },
    "capital": {
        "patterns": ["What is the capital of France?", "Capital of France", "Where is the capital of France?", "France's capital city"],
        "responses": ["The capital of France is Paris."]
    },
    "area": {
        "patterns": ["What is the total area of Canada?", "Area of Canada", "How big is Canada?", "Canada's total land area"],
        "responses": ["Canada has a total area of approximately 9.98 million square kilometers."]
    },
    "highest_mountain": {
        "patterns": ["What is the tallest mountain in the world?", "Highest mountain on Earth", "Name the highest peak on the planet", "World's tallest mountain"],
        "responses": ["The tallest mountain in the world is Mount Everest, with a height of approximately 8,848.86 meters."]
    },
    "deepest_ocean": {
        "patterns": ["What is the deepest ocean on Earth?", "Deepest ocean in the world", "Name the deepest part of the ocean", "Which ocean has the greatest depth?"],
        "responses": ["The deepest ocean in the world is the Pacific Ocean, with the Mariana Trench being the deepest part, reaching depths of approximately 10,994 meters."]
    },
    "largest_desert": {
        "patterns": ["What is the largest desert on Earth?", "Biggest desert in the world", "Name the largest arid region on the planet", "Which desert is the largest?"],
        "responses": ["The largest desert in the world is the Sahara Desert, covering an area of approximately 9.2 million square kilometers."]
    },
    "power_of_entry": {
        "patterns": [
            "What are the powers of entry, inspection, search, and seizure?",
            "Can officers enter and inspect mines?",
            "What can authorized officers do regarding mines?",
            "Explain the authority to search and seize in mines."
        ],
        "responses": [
            "Under the Act, authorized officers have various powers:",
            "- They can enter and inspect mines at reasonable times.",
            "- They can weigh, draw samples, and take measurements of mineral stocks.",
            "- They're authorized to survey, take samples, and measurements in mines.",
            "- Officers can examine documents, books, registers, and records related to mines.",
            "- They have the authority to order the production of documents.",
            "- Officers can also examine individuals connected with the mines."
        ]
    },
    "search_and_seizure": {
        "patterns": [
            "What are the procedures for search and seizure?",
            "Can officers search mines without a warrant?",
            "Explain the seizure of vessels and mines.",
            "How can officers enforce search and seizure?"
        ],
        "responses": [
            "Regarding search and seizure:",
            "- Officers can search mines without a warrant to ascertain compliance with the Act.",
            "- They can stop or board vessels engaged in regulated activities.",
            "- Officers can seize vessels, mines, equipment, and minerals involved in violations.",
            "- They're empowered to arrest individuals committing violations."
        ]
    },
    "offences": {
        "patterns": [
            "What are the offences under the Act?",
            "What penalties apply for violations?",
            "Explain the penalties for obstructing officers.",
            "Can companies be held liable for offences?"
        ],
        "responses": [
            "The Act specifies various offences and penalties:",
            "- Undertaking operations without necessary permits or licences.",
            "- Failing to provide required data or obstructing authorized officers.",
            "- Penalties include fines, imprisonment, and confiscation of vessels and minerals.",
            "- Companies and individuals can be held liable for violations."
        ]
    },
    "civil_liability": {
        "patterns": [
            "How is civil liability determined?",
            "Explain the liability for contravening terms and conditions.",
            "Who has jurisdiction over civil liability cases?",
            "What powers do authorized officers have for civil liability?"
        ],
        "responses": [
            "Civil liability is determined as follows:",
            "- Contravention of general and particular terms and conditions incurs liability.",
            "- Only authorized officers designated by the Central Government have jurisdiction.",
            "- Officers can file applications against permittees, licensees, or lessees for civil wrongs.",
            "- Authorized officers have powers similar to civil courts for adjudication."
        ]
    },
    "extension_of_enactments": {
        "patterns": [
            "Can enactments be extended to offshore areas?",
            "How are Indian enactments applied to offshore areas?",
            "Explain the extension of laws to offshore areas."
        ],
        "responses": [
            "Enactments can be extended to offshore areas as follows:",
            "- The Central Government can extend existing laws with restrictions and modifications.",
            "- Provisions are made for enforcement of such laws in offshore areas.",
            "- Extended enactments have the same effect as if the area were part of India."
        ]
    },
    "compounding_of_offences": {
        "patterns": [
            "Is there provision for compounding offences?",
            "Can offences under the Act be compounded?",
            "Explain how offences can be compounded."
        ],
        "responses": [
            "Offences under the Act can be compounded as follows:",
            "- Administering authorities or authorized officers can compound offences.",
            "- Offenders can pay a specified sum, not exceeding the maximum fine, to compound the offence.",
            "- Compounding an offence prevents further legal proceedings against the offender."
        ]
    },
    "recovery_of_sums": {
        "patterns": [
            "How are sums due to the Central Government recovered?",
            "Explain the recovery of licence fees and royalties.",
            "Can overdue amounts be recovered as arrears of land revenue?"
        ],
        "responses": [
            "Sums due to the Central Government are recovered as follows:",
            "- Licence fees, royalties, and other sums can be recovered as arrears of land revenue.",
            "- The administering authority can issue certificates for recovery.",
            "- Recovered sums, along with interest, are given priority over other assets."
        ]
    },
    "delegation_of_powers": {
        "patterns": [
            "Is there provision for delegation of powers?",
            "Can the Central Government delegate its powers?",
            "Explain how powers are delegated under the Act."
        ],
        "responses": [
            "Powers under the Act can be delegated as follows:",
            "- The Central Government can delegate its powers to subordinate officers or authorities.",
            "- Delegation is subject to specified conditions and matters.",
            "- Delegated officers or authorities can exercise powers in relation to designated matters."
        ]
    },
    "protection_of_action": {
        "patterns": [
            "Are there provisions for protecting actions taken in good faith?",
            "Is there protection against legal proceedings for actions under the Act?",
            "Explain the protection provided for good faith actions."
        ],
        "responses": [
            "Actions taken in good faith are protected as follows:",
            "- No legal proceedings can be initiated against individuals for actions done in good faith.",
            "- Protection extends to actions under the Act and its rules.",
            "- This provision safeguards individuals acting in accordance with the Act."
        ]
    },
    "mineral_rates": {
        "patterns": ["royalty rates", "fixed rent rates", "mineral rates", "rates of royalty", "rates of fixed rent", "mining rates"],
        "responses": [
            "The rates of royalty and fixed rent are provided below:\n\nRoyalty Rates:\n1. Brown ilmenite (leucoxene), Ilmenite, Rutile and Zircon: 2% of sale price on ad valorem basis.\n2. Dolomite: ₹40 per tonne.\n3. Garnet: 3% of sale price on ad valorem basis.\n4. Gold: 1.5% of London Bullion Market Association price.\n5. Limestone and Lime mud: ₹40 per tonne.\n6. Manganese Ore: 3% of sale price on ad valorem basis.\n7. Monazite: ₹125 per tonne.\n8. Sillimanite: 2.5% of sale price on ad valorem basis.\n9. Silver: 5% of London Metal Exchange price.\n10. All other minerals: 10% of sale price on ad valorem basis.",
            "Fixed Rent Rates:\nRates are provided in rupees per standard block per annum, varying based on the year of the lease.\n1st Year: [rate]\n2nd to 5th Year: [rate]\n6th to 10th Year: [rate]\n11th Year onwards: [rate]"
        ]
    },
    "notification_details": {
        "patterns": ["notification details", "S.O. 575(E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
        "responses": ["The notification S.O. 575(E) was issued by the Ministry of Mines, Government of India, on February 3rd, 2023.", "The purpose of the notification is to confer powers to the Jharkhand Exploration and Mining Corporation Limited, Ranchi, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is the Jharkhand Exploration and Mining Corporation Limited, Ranchi.", "The notification imposes a condition that the corporation must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "S.O. 208(E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
        "responses": ["The notification S.O. 208(E) was issued by the Ministry of Mines, Government of India, on January 12th, 2023.", "The purpose of the notification is to confer powers to M/s. FCI Aravali Gypsum and Minerals India Limited, a Central Government Company, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is M/s. FCI Aravali Gypsum and Minerals India Limited.", "The notification imposes a condition that the company must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 425 (E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
        "responses": ["The notification G.S.R. 425 (E) was issued by the Ministry of Mines, Government of India, on June 22nd, 2021.", "The purpose of the notification is to confer powers to Hutti Gold Mines Company Limited, Karnataka, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is Hutti Gold Mines Company Limited.", "The notification imposes a condition that the company must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
        "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
        "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
        "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 707(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
        "responses": ["The notification G.S.R. 707(E) was issued by the Ministry of Mines, Government of India, on July 27th, 2018.", "The purpose of the notification is to notify Hindustan Copper Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "Hindustan Copper Limited is the entity notified in this notification.", "The notification imposes a condition that Hindustan Copper Limited must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 389(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
        "responses": ["The notification G.S.R. 389(E) was issued by the Ministry of Mines, Government of India, on April 23rd, 2018.", "The purpose of the notification is to notify Odisha Mineral Exploration Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "Odisha Mineral Exploration Corporation Limited is the entity notified in this notification.", "The notification imposes a condition that Odisha Mineral Exploration Corporation Limited must share the data generated by prospecting operations with the State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 40(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
        "responses": ["The notification G.S.R. 40(E) was issued by the Ministry of Mines, Government of India, on January 18th, 2018.", "The purpose of the notification is to notify National Thermal Power Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "National Thermal Power Corporation Limited is the entity notified in this notification.", "The notification imposes a condition that National Thermal Power Corporation Limited must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "notification_details": {
        "patterns": ["notification details", "G.S.R. 1325(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
        "responses": ["The notification G.S.R. 1325(E) was issued by the Ministry of Mines, Government of India, on October 24th, 2017.", "The purpose of the notification is to notify M/s. Odisha Mining Corporation Limited and M/s. West Bengal Mineral Development and Trading Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "M/s. Odisha Mining Corporation Limited and M/s. West Bengal Mineral Development and Trading Corporation Limited are the entities notified in this notification.", "The notification imposes a condition that the notified entities must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
    },
    "mining_notifications": {
      "patterns": [
        "mining notification",
        "Mines and Minerals Act notification",
        "Ministry of Mines notification",
        "MMDR Act notification"
      ],
      "responses": [
        "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
      ]
    },
    "notification_details": {
      "patterns": [
        "details about notification",
        "details of notification",
        "notification details",
        "more info about notification",
        "what does the notification say"
      ],
      "responses": [
        "Of course! Please provide the notification number or the date of the notification you want details about."
      ]
    },
    "mining_notifications": {
      "patterns": [
        "mining notification",
        "Mines and Minerals Act notification",
        "Ministry of Mines notification",
        "MMDR Act notification"
      ],
      "responses": [
        "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
      ]
    },
    "notification_details": {
      "patterns": [
        "details about notification",
        "details of notification",
        "notification details",
        "more info about notification",
        "what does the notification say"
      ],
      "responses": [
        "Of course! Please provide the notification number or the date of the notification you want details about."
      ]
    },
    "notification_2015_07_06": {
      "patterns": [
        "notification dated 6th July 2015",
        "Mines and Minerals Act notification G.S.R. 538(E)"
      ],
      "responses": [
        "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
      ]
    },
    "mining_notifications": {
      "patterns": [
        "mining notification",
        "Mines and Minerals Act notification",
        "Ministry of Mines notification",
        "MMDR Act notification"
      ],
      "responses": [
        "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
      ]
    },
    "notification_details": {
      "patterns": [
        "details about notification",
        "details of notification",
        "notification details",
        "more info about notification",
        "what does the notification say"
      ],
      "responses": [
        "Of course! Please provide the notification number or the date of the notification you want details about."
      ]
    },
    "notification_2015_07_06": {
      "patterns": [
        "notification dated 6th July 2015",
        "Mines and Minerals Act notification G.S.R. 538(E)"
      ],
      "responses": [
        "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
      ]
    },
    "mining_notifications": {
      "patterns": [
        "mining notification",
        "Mines and Minerals Act notification",
        "Ministry of Mines notification",
        "MMDR Act notification"
      ],
      "responses": [
        "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
      ]
    },
    "notification_details": {
      "patterns": [
        "details about notification",
        "details of notification",
        "notification details",
        "more info about notification",
        "what does the notification say"
      ],
      "responses": [
        "Of course! Please provide the notification number or the date of the notification you want details about."
      ]
    },
    "notification_2015_07_06": {
      "patterns": [
        "notification dated 6th July 2015",
        "Mines and Minerals Act notification G.S.R. 538(E)"
      ],
      "responses": [
        "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
      ]
    },
    "mining_notifications": {
      "patterns": [
        "mining notification",
        "Mines and Minerals Act notification",
        "Ministry of Mines notification",
        "MMDR Act notification"
      ],
      "responses": [
        "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
      ]
    },
    "notification_details": {
      "patterns": [
        "details about notification",
        "details of notification",
        "notification details",
        "more info about notification",
        "what does the notification say"
      ],
      "responses": [
        "Of course! Please provide the notification number or the date of the notification you want details about."
      ]
    },
    "notification_2015_07_06": {
      "patterns": [
        "notification dated 6th July 2015",
        "Mines and Minerals Act notification G.S.R. 538(E)"
      ],
      "responses": [
        "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
      ]
    },
    "PMKKKY_Objective": {
        "patterns": ["What is the objective of PMKKKY?", "What are the objectives of Pradhan Mantri Khanij Kshetra Kalyan Yojana?", "What does PMKKKY aim to achieve?"],
        "responses": ["The objective of PMKKKY is to implement developmental and welfare projects/programs in mining affected areas, mitigate adverse impacts of mining on environment and socio-economics, and ensure sustainable livelihoods for affected people."]
    },
    "DMF_Establishment": {
        "patterns": ["When was the District Mineral Foundation established?", "What is District Mineral Foundation?", "When was DMF established?"],
        "responses": ["The District Mineral Foundation (DMF) was established in all districts affected by mining operations through the amendment of the Mines & Minerals (Development & Regulation) Act, 1957, in 2015."]
    },
    "DMF_Funding": {
        "patterns": ["How is the District Mineral Foundation funded?", "What are the sources of funding for DMF?", "Who funds the DMFs?"],
        "responses": ["DMFs are funded by statutory contributions from mining lease holders."]
    },
    "PMKKKY_Implementation": {
        "patterns": ["How is PMKKKY implemented?", "Who implements Pradhan Mantri Khanij Kshetra Kalyan Yojana?"],
        "responses": ["PMKKKY is implemented by the District Mineral Foundations (DMFs) of the respective districts using the funds accruing to the DMF."]
    },
    "PMKKKY_Composition": {
        "patterns": ["What is the composition of PMKKKY?", "Who is included in the composition of PMKKKY?"],
        "responses": ["The composition of PMKKKY includes MPs, MLAs, and MLCs in the Governing Council, as per the directive of the Central Government."]
    },
    "PMKKKY_Directions": {
        "patterns": ["What directions were issued regarding PMKKKY?", "What directions did the Central Government issue regarding PMKKKY?"],
        "responses": ["The Central Government issued directions regarding composition and utilization of funds by DMFs, including the inclusion of MPs, MLAs, and MLCs in the Governing Council, prohibition of fund transfers to state exchequer, and preparation of a five-year Perspective Plan."]
    },
    "PMKKKY_Revision": {
        "patterns": ["Are there any revised PMKKKY guidelines?", "What are the revised PMKKKY guidelines?"],
        "responses": ["Yes, revised PMKKKY guidelines have been issued under section 9B (3) of the MMDR Act 1957 after consultation with stakeholders."]
    },
    "Affected_Areas_People": {
        "patterns": ["How are affected areas and people identified under PMKKKY?", "What is the process of identifying affected areas and people under PMKKKY?"],
        "responses": ["Affected areas and people are identified based on direct and indirect impacts of mining operations, including displacement, economic dependence, and environmental consequences."]
    },
    "Utilization_of_Funds": {
        "patterns": ["How are the funds under PMKKKY utilized?", "What is the utilization of funds under PMKKKY?"],
        "responses": ["PMKKKY funds are utilized for high priority sectors such as drinking water supply, healthcare, education, livelihood generation, and environment preservation, with a minimum allocation of 70% to directly affected areas."]
    },
    "composition_and_functions": {
        "patterns": "The Chairman of Governing Council and Managing Committee shall be the District Magistrate/Deputy Commissioner/Collector of the district. No other person shall function as Chairman of the Governing Council and/or Managing Committee.",
        "responses": {
            "mp": "MPs representing mining-affected areas shall be members of the Governing Council. Each Lok Sabha MP with mining-affected areas in their constituency shall be a member. If an MP's constituency spans multiple districts, they shall be a member of the Governing Council in each district. Similarly, a Rajya Sabha MP shall be a member of the Governing Council of one district.",
            "mla": "MLAs representing mining-affected areas shall be members of the Governing Council. If an MLA's constituency spans multiple districts, they shall be a member of the Governing Council in each district.",
            "mlc": "Members of Legislative Council shall be a member of the Governing Council of one district chosen by them.",
            "meetings": "The Governing Council shall convene at least twice a year, with meeting dates scheduled according to the convenience of MP members.",
            "managing_committee": "The Managing Committee shall consist of the District Magistrate/Deputy Commissioner/Collector as Chairman and senior district officers responsible for project execution. Elected representatives or nominated non-official members are not part of the committee.",
            "meeting_frequency": "The Managing Committee shall meet at least once every quarter."
        }
    },
    "introduction": {
        "patterns": ["what is the introduction?", "introduction", "purpose of the committee", "committee's purpose"],
        "response": "The committee was formed to examine the issue of misclassification of grades of iron ore and other minerals and to suggest measures for preventing it. It was also tasked with exploring the adoption of advanced technology in this regard."
    },
    "methodology": {
        "patterns": ["what methodology was adopted?", "methodology adopted by the committee", "how did the committee conduct its study?"],
        "response": "The committee adopted a consultative approach, including presentations by the Indian Bureau of Mines (IBM) and state government representatives. Written comments were invited from industry associations, companies, and other stakeholders. A sub-committee visited iron ore mines in Odisha to understand sampling issues and explore new technologies. Thirteen meetings were held to discuss the findings and recommendations."
    },
    "sampling_dispatch_transportation": {
        "patterns": ["what is the existing procedure of sampling, dispatch, and transportation?", "existing procedure of sampling", "existing procedure of dispatch", "existing procedure of transportation"],
        "response": "State governments have established rules to prevent illegal mining, transportation, and storage of minerals, including measures such as check-posts and weigh-bridges. After excavation, minerals are sorted into different sizes and grades and dispatched to end-users based on demand. Online systems are used for royalty assessment and payment, with weighment done at mine sites. Penalties are prescribed for transporting minerals without lawful authority."
    },
    "average_sale_price": {
        "patterns": ["what is the average sale price (ASP)?", "relevance of ASP", "how is ASP calculated?", "ASP calculation"],
        "response": "ASP is crucial for calculating revenue to state governments, particularly in lease auctions. It is used for royalty assessment, valuation of mineral blocks for auctions, and calculating bid premiums. Online returns are filed by mining lease holders, and ASP compilation is based on these returns."
    },
    "iron_ore_misclassification": {
        "patterns": ["impact of misclassification", "effect of misreporting", "revenue loss due to misclassification", "state government concerns", "iron ore grade classification", "measures to prevent misclassification", "technological solutions for misclassification"],
        "responses": ["Misclassification of iron ore grades can have significant financial implications, particularly for state governments and other stakeholders involved in the mining and mineral industry. It can lead to revenue losses due to differences in ASP and impact revenue generation through royalty, auction premiums, etc. State governments are concerned about mine owners replacing higher grades of iron ore with lower grades in their reports, which directly affects revenue generation. Measures to address misclassification include implementing robust mechanisms for sampling and grade declaration to prevent revenue losses. Some states have adopted IT-based systems for sampling and analysis. Technological solutions such as handheld XRF analyzers and cryptography for data verification during transportation are also being considered."],
    },
    "subcommittee_constitution": {
        "patterns": ["constitution of sub-committee", "formation of sub-committee", "sub-committee visit to Odisha"],
        "responses": ["To understand the issues involved in sampling and declaration of grades of iron ore and explore new technologies, a sub-committee was formed. The members visited Odisha from 31.05.2022 to 02.06.2022."],
    },
    "subcommittee_members": {
        "patterns": ["members of the sub-committee", "who participated in the tour"],
        "responses": ["The members of the sub-committee were: 1) Sh. Dheeraj Kumar, Deputy Secretary (Mines) 2) Sh. S.K. Adhikari, CMG, IBM 3) Sh. Sanjay Khare, Dy. Director, Govt. of Chhattisgarh (nominated by the State Government) 4) Sh. Salil Behera, Jt. Director of Mines, Govt. of Odisha (nominated by the State Government) 5) Sh. Sambhav Jain, Sr. Manager(Legal), NALCO Ltd./ Ministry of Mines"],
    },
    "visit_details": {
        "patterns": ["details of the visit", "purpose of the visit", "observations from the visit"],
        "responses": ["The sub-committee visited Odisha to understand sampling and grade declaration issues. They visited Joda East Iron Ore Mine (exempted) and Jajang Iron Ore Mine (non-exempted). They noted shortcomings in the stacking, sampling, and analysis process."],
    },
    "sampling_technology": {
        "patterns": ["presentation by technology suppliers", "latest sampling technology", "auto-sampling and sample analysis"],
        "responses": ["The committee explored technologies for iron ore grading. They considered options like cross-belt conveyor sampling system, auger system, and laser analyzer. The laser technology was found promising to resolve misclassification issues."],
    },
    "mineral_recommendations": {
        "patterns": ["recommendations", "mineral recommendations", "what are the recommendations"],
        "response": "The recommendations include implementing an IT-enabled system for sampling, analysis, and transportation monitoring, integration of leaseholder systems with government monitoring, incentives for adopting new technologies, use of IT-based grade information systems, automated sampling and analysis, videography of sampling process, random sampling, and regular audits."
    },
    "state_government_powers": {
        "patterns": ["state government powers", "powers of state governments", "what are the powers of state governments"],
        "response": "State governments have complete legislative and administrative powers related to transportation and storage of minerals. They also receive revenue from mineral production in the form of royalty, auction premium, and other payments."
    },
    "penal_action": {
        "patterns": ["penal action", "what happens in case of illegal transportation", "what are the penalties for illegal transportation"],
        "response": "State governments have sufficient powers to take penal action against illegal transportation of minerals. Any mineral transported in contravention of state government rules shall be considered 'without lawful authority' and attract penalties as prescribed under the MMDR Act."
    },
    "recommendations_implementation": {
        "patterns": ["implementation of recommendations", "how to implement recommendations", "what is the process to implement recommendations"],
        "response": "Recommendations should be implemented by state governments through rules under section 23C of the MMDR Act and other guidelines. The central government may issue necessary advice to facilitate implementation and maintain uniformity in rules across states."
    },
    "misclassification_of_minerals": {
        "patterns": ["misclassification of minerals", "how to prevent misclassification", "recommendations for preventing misclassification"],
        "response": "To prevent misclassification of minerals, it is recommended to implement an IT-enabled system with minimal human intervention, use of technologies like continuous online analyzers for sampling and analysis, videography of sampling process, random sampling, and regular audits."
    },
    "transportation_recommendations": {
        "patterns": ["transportation recommendations", "how to improve mineral transportation", "recommendations for mineral transportation"],
        "response": "Recommendations for mineral transportation include the use of GPS-enabled vehicles with RFID tagging for monitoring, pre-registration of mineral-carrying vehicles with government portals, and establishment of mine monitoring systems with geo-fencing."
    },
    "blockchain_use_in_mining": {
        "patterns": ["blockchain use in mining", "how can blockchain help in mining", "benefits of blockchain in mining"],
        "response": "Blockchain technology can enhance transparency, efficiency, and security in the mining industry by enabling real-time tracking of value chains and supply chains, facilitating self-declaration of grades, tracking materials from extraction to production, automating invoice reconciliation, improving traceability of reserves, and validating workflow/audit processes."
    },
    "blockchain_recommendations": {
        "patterns": ["blockchain recommendations", "how to implement blockchain in mining", "recommendations for adopting blockchain in mining"],
        "response": "The committee recommends the adoption of blockchain technology in the mining sector, starting with a pilot project for high-value minerals like gold, copper, zinc, etc. Learnings from the pilot project can inform the replication of the model in other mines and minerals."
    },
    "grade_drop_in_odisha": {
        "patterns": ["grade drop in Odisha", "reasons for grade drop in Odisha", "observations on grade drop in Odisha"],
        "response": "The committee observed a sharp drop in the grade of ores produced in Odisha after March 31, 2021. This warrants detailed study of field operations, and concerned state governments should refer such cases to the central government for further investigation by authorities like the Geological Survey of India."
    },
    "applicability_on_other_ores": {
        "patterns": ["applicability on other ores", "how do recommendations apply to other ores", "impact on other ores"],
        "response": "The recommendations provided, mainly for iron ore, can be applied to all other ores where royalty, auction premium, and other payments to the government are dependent on the grade of the ore."
    },
    "views_of_state_governments": {
        "patterns": ["views of state governments on the final draft", "state governments' opinions on the final report", "feedback from state governments on the final draft"],
        "response": "State governments have provided feedback on the final draft of the report:\n- Chhattisgarh: Appreciates the recommendations and agrees with them, subject to consultation with stakeholders for feasibility. Already implementing certain measures like Khanij Online system.\n- Karnataka: Agreed with the final draft.\n- Odisha: Provided observations on the final draft.\n- Jharkhand: Agreed with the final draft."
    },
    "famous_scientists": {
        "patterns": ["Who are some famous scientists?", "Can you name some renowned scientists?", "Tell me about famous scientists.", "Give me examples of notable scientists.", "Which scientists are well-known?"],
        "responses": ["Some famous scientists include Albert Einstein, Isaac Newton, Marie Curie, Charles Darwin, and Nikola Tesla.", "Renowned scientists include Stephen Hawking, Galileo Galilei, Ada Lovelace, and Richard Feynman.", "Famous scientists throughout history include Leonardo da Vinci, Thomas Edison, Jane Goodall, and Neil deGrasse Tyson."]
    },
    "capital_cities": {
        "patterns": ["What are the capital cities of different countries?", "Tell me about capital cities around the world.", "Can you list the capitals of various countries?", "What are the capitals of some nations?", "Give me information about capital cities."],
        "responses": ["Some capital cities include London (United Kingdom), Paris (France), Tokyo (Japan), Beijing (China), and Moscow (Russia).", "Capital cities around the world include Washington, D.C. (United States), Berlin (Germany), Rome (Italy), Brasília (Brazil), and Canberra (Australia).", "The capitals of various countries include New Delhi (India), Ottawa (Canada), Cairo (Egypt), Buenos Aires (Argentina), and Seoul (South Korea)."]
    },
  "sampling_process": {
    "patterns": ["What is the sampling process in Odisha mines?", "How does the sampling process work in Odisha mines?", "Can you explain the sampling process in Odisha mines?", "Tell me about the sampling process in non-exempted mines in Odisha."],
    "responses": ["In Odisha mines, the sampling process involves various steps such as stack creation, sample collection, and chemical analysis. It utilizes technology like mobile apps and augmented reality for accuracy and efficiency."]
  },
  "issues_observed": {
    "patterns": ["What are the issues observed in the stacking and sampling process?", "What challenges are faced in the stacking and sampling process?", "What problems have been identified in the stacking and sampling process?"],
    "responses": ["Several issues have been observed in the stacking and sampling process in Odisha mines, including challenges related to stack size, space requirements, sampling accuracy, and human intervention. The process is also time-consuming and lacks continuous monitoring."]
  },
  "technological_advancement": {
    "patterns": ["How can the stacking and sampling process be improved?", "What improvements can be made to the stacking and sampling process?", "Are there any suggestions for enhancing the stacking and sampling process?"],
    "responses": ["To improve the stacking and sampling process, there is a need for technological advancement and automation. This would streamline processes, enhance efficiency, and ensure compliance with regulatory requirements."]
  },
  "additional_note": {
    "pattern": "Additional note to the report dated 11.11.2022 of the committee on misclassification of grades of different grades of iron ore and other minerals –reg.",
    "responses": [
      "Ministry of Mines constituted a committee to examine the issue of misclassification of grades of iron ore and other minerals, adversely affecting the revenue of State Governments and suggest measures for preventing misclassification.",
      "Comments/suggestions of stakeholders were sought on the recommendations of the committee. Comments received were forwarded to the committee for further consideration.",
      "A meeting of the committee was convened on 13.07.2023 to discuss the comments received.",
      "The committee observed that State Governments of Jharkhand, Chhattisgarh, and Karnataka generally accepted/agreed to the recommendations, while the State Government of Odisha had existing IT-based systems.",
      "Recommendations in the report are recommendatory, and it is up to the State Governments to implement them.",
      "The committee suggests implementing its recommendations on a pilot basis and utilizing learnings to prevent misreporting/misclassification of grades of minerals.",
      "Additional recommendations suggested by the committee:",
      "1. Adequate redressal mechanism for variation between physical inspection and system-enabled results.",
      "2. Provision to allow dispatch in case of non-functionality of any equipment in the entire system.",
      "3. The recommendation of the committee can be of recommendatory nature, and the adoption can be left to the State Government.",
      "4. The system may initially be implemented on a pilot basis through a PSU."
    ]
  },
    "misclassification_guidelines": {
        "patterns": ["misclassification guidelines", "guidelines for preventing misclassification", "iron ore misclassification prevention", "mineral misclassification prevention"],
        "responses": ["The guidelines aim to prevent misclassification of different grades of iron ore and other minerals, ensuring accurate assessment of Average Sale Price (ASP) and proper collection of statutory levies such as royalty."]
    },
    "mineral_sampling_analysis": {
        "patterns": ["mineral sampling and analysis", "sampling and analysis guidelines", "mineral grade determination", "mineral analysis process"],
        "responses": ["The guidelines emphasize the adoption of IT-enabled systems with minimal human intervention for sampling and analysis, integration of leaseholder systems with government monitoring systems, and encouragement of new technologies adoption through incentives."]
    },
    "continuous_online_analysis": {
        "patterns": ["continuous online analysis", "online analyzer installation", "real-time analysis monitoring", "augur-based auto-samplers"],
        "responses": ["Mandatory installation of continuous online analyzers for large mines, both mechanized and non-mechanized, use of augur-based auto-samplers for non-mechanized loading systems, and real-time monitoring of analysis results and CCTV surveillance are recommended."]
    },
    "royalty_statutory_payments": {
        "patterns": ["royalty and statutory payments", "payment of royalty", "mineral dispatch payment", "seam analysis for royalty"],
        "responses": ["Charging of royalty based on continuous analyzer analysis, monthly seam analysis for tolerance limit determination, and collection of other statutory payments are recommended."]
    },
    "transportation_monitoring": {
        "patterns": ["transportation monitoring", "GPS-enabled vehicles", "RFID tagging", "vehicle tracking", "geo-fencing of mines"],
        "responses": ["Use of GPS-enabled vehicles with RFID tagging for tracking and monitoring, pre-registration of mineral-carrying vehicles, and geo-fencing of mine boundaries are recommended."]
    },
    "blockchain_technology": {
        "patterns": ["blockchain technology", "blockchain for mineral tracking", "blockchain for transparency", "blockchain for accountability"],
        "responses": ["Implementation of blockchain for transparent tracking of mineral transactions, self-declaration of grades, invoice reconciliation, and potential pilot projects for high-value minerals are recommended."]
    },
    "applicability_to_other_ores": {
        "patterns": ["applicability to other ores", "guidelines for other minerals", "royalty guidelines for other minerals", "grade-based payments for other ores"],
        "responses": ["The guidelines are applicable to all ores where payments to the government depend on ore grade."]
    },
    "additional_recommendations": {
        "patterns": ["additional recommendations", "misclassification prevention suggestions", "redressal mechanism", "dispatch provision for equipment failure"],
        "responses": ["Establishment of a redressal mechanism for variation in inspection results, provision for dispatch in case of equipment non-functionality, and pilot implementation through a PSU with cross-verification of accuracy and detection limits are recommended."]
    },






     "mineral_rules": {
        "patterns": [
            "What are the Mineral Conservation and Development Rules?",
            "Tell me about the Mineral Conservation and Development Rules, 2017.",
            "Explain the amendments made in the Mineral Conservation and Development Rules, 2024.",
            "When do the Mineral Conservation and Development (Amendment) Rules, 2024 come into force?",
            "What is mentioned in rule 4 of the Mineral Conservation and Development Rules, 2017?",
            "Can you provide information on reconnaissance permits and prospecting licenses?",
            "How long does a holder of a reconnaissance permit or prospecting license have to submit a scheme?",
            "Tell me about the modifications required for exploration licenses.",
            "What is sub-section (11) of section 10BA in the Mineral Conservation and Development Rules?"
        ],
        "responses": [
            "The Mineral Conservation and Development Rules govern the mining sector. The latest amendments are in the Mineral Conservation and Development (Amendment) Rules, 2024.",
            "The Mineral Conservation and Development Rules, 2017, were amended by the Mineral Conservation and Development (Amendment) Rules, 2024.",
            "The Mineral Conservation and Development (Amendment) Rules, 2024, came into force on the date of their publication in the Official Gazette.",
            "In rule 4 of the Mineral Conservation and Development Rules, 2017, changes include the submission of a scheme for reconnaissance or prospecting within ninety days.",
            "Reconnaissance permit and prospecting license holders must submit a scheme for operations within ninety days of obtaining the permit or license.",
            "For exploration licenses, a modified scheme must be submitted after three years, indicating how the licensee plans to continue operations in the retained area.",
            "Sub-section (11) of section 10BA relates to the retention of areas under exploration licenses.",
        ]
    },

"mineral_rules_amendments": {
        "patterns": [
            "What are the Mineral (Auction) Amendment Rules, 2024?",
            "Tell me about the recent amendments in the Mineral (Auction) Rules.",
            "When do the Mineral (Auction) Amendment Rules, 2024, come into force?",
            "Explain the changes in rule 2 of the Mineral (Auction) Rules, 2015.",
            "What is the definition of 'auction premium' in the amended rules?",
            "How is exploration license granted for minerals in the Seventh Schedule?",
            "Can a bidder submit more than one bid in an auction?",
            "Define 'affiliate' in the context of bidding.",
            "What is the impact of the amendments on upfront payment for preferred bidders?",
            "Tell me about the modifications in rule 19 regarding performance security.",
            "Is there a limit on the performance security for holders of composite licenses?",
            "What is mentioned in the new Rule 19A?"
        ],
        "responses": [
            "The Mineral (Auction) Amendment Rules, 2024, are the recent amendments to the Mineral (Auction) Rules, 2015.",
            "The Mineral (Auction) Amendment Rules, 2024, come into force upon publication in the Official Gazette.",
            "Rule 2 of the Mineral (Auction) Rules, 2015, has been amended to include a definition of 'auction premium' and modify references in various clauses.",
            "In the amended rules, 'auction premium' is defined as the amount payable by the lessee under sub-rule (2) of rule 13.",
            "Exploration licenses for minerals in the Seventh Schedule are granted as specified in Chapter III A.",
            "No, a bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
            "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
            "For preferred bidders selected after the commencement of the Mineral (Auction) Amendment Rules, 2024, the upfront payment should not exceed five hundred crore rupees.",
            "Rule 19 has provisos specifying the maximum performance security amounts for preferred bidders and holders of composite licenses.",
            "For preferred bidders selected after the commencement of the Mineral (Auction) Amendment Rules, 2024, the performance security limit should not exceed two hundred and fifty crore rupees.",
            "A new rule, Rule 19A, has been introduced after Rule 19."
        ]
    },
 "exploration_license": {
        "patterns": [
            "What is the process for obtaining an exploration license?",
            "Tell me about the auction process for exploration licenses.",
            "What are the prerequisites for initiating the auction of an exploration license?",
            "Explain the role of the committee in identifying blocks for auction.",
            "How does the State Government decide on areas for auction?",
            "Can individuals submit proposals for exploration licenses?",
            "What is the eligibility criteria for participating in the auction?",
            "How is eligibility determined for exploration license bids?",
            "Can a bidder submit more than one bid in an auction?",
            "Define 'affiliate' in the context of exploration license bidding.",
            "What are the exclusion criteria for identifying blocks for auction?",
            "How does the Central Government approve auction recommendations?",
            "Tell me about the requirements for participating in exploration license auctions.",
            "What is the significance of Schedule I in the exploration license process?",
            "Define terms like 'associate company,' 'control,' and 'subsidiary company' as per the Companies Act, 2013."
        ],
        "responses": [
            "The process for obtaining an exploration license involves an auction initiated by the State Government.",
            "Prerequisites for initiating the auction of an exploration license include submitting proposals and available geoscience data.",
            "A committee, including members from various departments, identifies blocks for auction based on geological information.",
            "The State Government decides on areas for auction through the committee, considering geological reports and data repositories.",
            "Yes, individuals can submit proposals for exploration licenses by following the format specified in Schedule V.",
            "Eligibility for participating in the auction is determined based on section 5 requirements and terms in Schedule I.",
            "A bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
            "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
            "Exclusion criteria for identifying blocks include areas with existing concessions, ongoing tender processes, or certain exploration operations.",
            "The Central Government approves auction recommendations within a specified time frame.",
            "Requirements for participating in exploration license auctions are outlined in section 5 and Schedule I.",
            "Schedule I specifies terms and conditions for eligibility and participation in exploration license auctions.",
            "'Associate company,' 'control,' and 'subsidiary company' have the same meanings as assigned in the Companies Act, 2013."
        ]
    },
 "exploration_license": {
        "patterns": [
            "What is the process for obtaining an exploration license?",
            "How is the auction process for exploration licenses conducted?",
            "Tell me about the prerequisites for initiating an exploration license auction.",
            "Explain the role of the committee in identifying blocks for auction.",
            "How does the State Government decide on areas for auction?",
            "Can individuals submit proposals for exploration licenses?",
            "What are the eligibility criteria for participating in the auction?",
            "How is eligibility determined for exploration license bids?",
            "Can a bidder submit more than one bid in an exploration license auction?",
            "Define 'affiliate' in the context of exploration license bidding.",
            "What are the exclusion criteria for identifying blocks for auction?",
            "How does the Central Government approve auction recommendations?",
            "Tell me about the requirements for participating in exploration license auctions.",
            "What is the significance of Schedule I in the exploration license process?",
            "Define terms like 'associate company,' 'control,' and 'subsidiary company' as per the Companies Act, 2013."
        ],
        "responses": [
            "The process for obtaining an exploration license involves an auction initiated by the State Government.",
            "Prerequisites for initiating the auction of an exploration license include submitting proposals and available geoscience data.",
            "A committee, including members from various departments, identifies blocks for auction based on geological information.",
            "The State Government decides on areas for auction through the committee, considering geological reports and data repositories.",
            "Yes, individuals can submit proposals for exploration licenses by following the format specified in Schedule V.",
            "Eligibility for participating in the auction is determined based on section 5 requirements and terms in Schedule I.",
            "A bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
            "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
            "Exclusion criteria for identifying blocks include areas with existing concessions, ongoing tender processes, or certain exploration operations.",
            "The Central Government approves auction recommendations within a specified time frame.",
            "Requirements for participating in exploration license auctions are outlined in section 5 and Schedule I.",
            "Schedule I specifies terms and conditions for eligibility and participation in exploration license auctions.",
            "'Associate company,' 'control,' and 'subsidiary company' have the same meanings as assigned in the Companies Act, 2013."
        ]
    },
    "electronic_auction": {
        "patterns": [
            "How is an auction conducted for exploration licenses electronically?",
            "Tell me about the platform used for electronic exploration license auctions.",
            "What are the bidding parameters for exploration licenses?",
            "Explain the electronic auction process for exploration licenses.",
            "How does the State Government specify the ceiling price for auction premiums?",
            "What is the bidding process for exploration licenses?",
            "Tell me about the notice inviting tender for exploration license auctions.",
            "What are the requirements for submitting a technical bid in exploration license auctions?",
            "Define bid security and its amount for exploration license auctions.",
            "How are technically qualified bidders determined in exploration license auctions?",
            "What is the process for the second round of online electronic auction?",
            "How does the Central Government conduct auctions for exploration licenses?",
            "What information does the State Government provide to the Central Government regarding exploration license auctions?",
            "How is the preferred bidder determined after a successful auction?"
        ],
        "responses": [
            "Exploration license auctions are conducted exclusively through online electronic auction platforms.",
            "State Governments can use online platforms meeting specified technical and security requirements.",
            "The State Government specifies a maximum percentage share ('ceiling price') of the auction premium for future lessees.",
            "The auction is a descending reverse online electronic auction with two rounds.",
            "Bidders quote a percentage share of the auction premium, and the one quoting the minimum percentage becomes the preferred bidder.",
            "State Government issues a notice inviting tender, providing details on the area under auction and available geoscience data.",
            "The tender document includes information on the identified area, geoscience data, and the bidding process.",
            "Bidders submit a technical bid and an initial price offer in the first round of the auction.",
            "Bid security is required based on the area size.",
            "Only technically qualified bidders proceed to the second round, submitting a final price offer.",
            "The lowest bidder in the second round becomes the preferred bidder.",
            "The State Government informs the Central Government about various stages of exploration license auctions.",
            "Central Government follows the same auction rules as applicable to State Governments for exploration licenses.",
            "Upon successful completion of the auction, the Central Government informs the State Government about the preferred bidder."
        ]
    },
 "exploration_license": {
        "patterns": ["What is an exploration license?", "How can I obtain an exploration license?", "Tell me about exploration licenses."],
        "responses": ["An exploration license allows individuals to initiate the process of obtaining mining rights for specified minerals. The process involves submitting a proposal to the State Government and participating in an auction."]
    },
    "grant_process": {
        "patterns": ["How is the exploration license granted?", "Explain the grant process for exploration licenses."],
        "responses": ["The exploration license grant process involves submitting a proposal, participating in an auction, and fulfilling conditions such as obtaining approvals and submitting a reconnaissance or prospecting scheme."]
    },
    "performance_security": {
        "patterns": ["What is performance security?", "Tell me about the performance security.", "Why is performance security required?"],
        "responses": ["Performance security is a financial guarantee provided by the preferred bidder. It ensures commitment to the exploration license process. It may be forfeited in case of non-compliance with specified conditions."]
    },
    "payment_process": {
        "patterns": ["How is payment handled for exploration licenses?", "Tell me about the payment process for exploration licenses."],
        "responses": ["Payment to exploration licensees involves receiving a percentage share from the auction premium deposited by the future lessee. The share is payable for the entire mining lease period or until resource exhaustion."]
    },
    "share_transfer": {
        "patterns": ["Can the exploration license share be transferred?", "Tell me about transferring the exploration license share."],
        "responses": ["Yes, the exploration license share can be transferred to another entity with State Government approval."]
    },
    "timeline_conditions": {
        "patterns": ["What are the timelines and conditions for exploration licenses?", "Explain the timelines and conditions during exploration licenses."],
        "responses": ["Timelines include submission of performance security, fulfillment of conditions, and surrender options. Conditions involve compliance, approvals, and submission of a prospecting scheme."]
    },
    "geological_exploration": {
        "patterns": ["What is geological exploration?", "Explain the geological exploration process during exploration licenses."],
        "responses": ["Geological exploration involves studying the mineral content of the licensed area. Exploration licensees submit periodic reports to the State Government and the Indian Bureau of Mines."]
    },
    "termination_conditions": {
        "patterns": ["Under what conditions can exploration licenses be terminated?", "Explain the conditions for exploration license termination."],
        "responses": ["Exploration licenses can be terminated if the licensee fails to complete operations or establish mineral contents within the specified period. The State Government may take appropriate actions."]
    },
 "exploration_license_auction": {
        "patterns": ["What are the rules for exploration license auction?", "Tell me about auction rules for exploration license.", "How is exploration license auction conducted?"],
        "responses": ["The auction for exploration license is conducted through an online electronic platform. The process involves various steps such as issuing tender documents, submitting bids, and a descending reverse online auction."]
    },
    "mining_lease_auction": {
        "patterns": ["How is mining lease auction conducted?", "Tell me about auction rules for mining lease.", "What are the terms for mining lease auction?"],
        "responses": ["Mining lease auction follows the rules specified in Chapter II of the regulations. The State Government initiates the auction within six months of receiving the geological report from the exploration licensee. The preferred bidder is selected within one year from the date of receiving the geological report."]
    },
    "preferred_bidder_selection": {
        "patterns": ["How is the preferred bidder selected?", "What factors determine the preferred bidder?", "Tell me about the selection of the preferred bidder."],
        "responses": ["The preferred bidder is selected based on various factors, including the geological report submitted by the exploration licensee. If the preferred bidder is not selected within the specified period, the State Government compensates the exploration licensee for the incurred expenditure."]
    },
    "termination_or_surrender": {
        "patterns": ["What happens in case of termination or surrender of a mining lease?", "Tell me about the consequences of lease termination.", "Explain the process if a lease is surrendered."],
        "responses": ["In case of termination, lapse, or surrender of a mining lease, the State Government provides an opportunity to the exploration licensee to obtain the lease in the same area at the auction premium discovered earlier."]
    },
    "participation_in_auction": {
        "patterns": ["Can the exploration licensee participate in the auction?", "Are there restrictions on participation in mining lease auction?", "Tell me about eligibility for auction participation."],
        "responses": ["The exploration licensee is not prohibited from participating in the auction for the mining lease. However, they need to fulfill the eligibility conditions specified in rule 6."]
    },
    "tender_document_details": {
        "patterns": ["What information is in the tender document?", "Tell me about the contents of the tender document.", "What details are provided in the auction tender document?"],
        "responses": ["The tender document contains details such as raw data and bore-hole cores generated during prospecting operations. Additionally, it includes ownership structure or shareholding details of the exploration licensee."]
    },
    "related_party_declaration": {
        "patterns": ["What is a related party declaration?", "Why is declaring related party important?", "Explain related party declaration in mining lease auction."],
        "responses": ["A bidder participating in the auction must declare to the government if they are a related party of the exploration licensee. This declaration ensures transparency in the auction process."]
    },
 "exploration_license": {
        "patterns": [
            "Tell me about exploration licenses",
            "What is the process for auctioning exploration licenses?",
            "How can I participate in the auction for exploration licenses?",
            "Explain the requirements for exploration license proposals",
            "Details needed for proposing an exploration license"
        ],
        "responses": [
            "Exploration licenses are granted for the prospecting of mineral-rich areas. The process involves submitting a proposal to the Mining and Geology Department.",
            "Auctioning exploration licenses follows specific rules. The State Government initiates the process within six months of receiving a geological report.",
            "To participate in the auction, you need to submit a proposal with details about the location, mineral potential, and relevant documentation.",
            "Requirements for exploration license proposals include applicant information, area details, mineral potential, and necessary documentation.",
            "For proposing an exploration license, you need to provide applicant details, location specifics, details of mineral potential, and required documentation."
        ]
    },
"amendment_rules": {
        "patterns": ["Tell me about the Mineral Conservation and Development (Amendment) Rules, 2024", "What are the changes in the rules?", "Explain the amendment rules"],
        "responses": [
            "The Mineral Conservation and Development (Amendment) Rules, 2024 were introduced by the Central Government to amend the Mineral Conservation and Development Rules, 2017.",
            "These rules came into force upon their publication in the Official Gazette.",
            "One significant change is in Rule 4, where the submission of schemes for reconnaissance or prospecting is now required within ninety days from the date of execution of the permit or license.",
            "In Rule 5, changes include the insertion of 'or both' after 'reconnaissance or prospecting' and the addition of exploration license in various places.",
            "Rule 9A imposes restrictions on the disclosure of information, schemes, and reports by the holder of an exploration license."
        ]
    },
    "submission_of_schemes": {
        "patterns": ["How should I submit a scheme for reconnaissance or prospecting?", "Tell me about the submission of schemes", "What is required for submitting a scheme?"],
        "responses": [
            "Every holder of a reconnaissance permit or prospecting license should submit a scheme within ninety days from the date of execution of the permit or license.",
            "For exploration license, a modified scheme should be submitted after three years from the date of execution.",
            "The scheme should indicate the manner in which the licensee proposes to carry out reconnaissance or prospecting operations in the covered area."
        ]
    },
    "half_yearly_reports": {
        "patterns": ["Explain the half-yearly reports", "What should be included in the half-yearly report?", "Tell me about reporting obligations"],
        "responses": [
            "Every holder of a reconnaissance permit, prospecting license, composite license, or exploration license should submit a half-yearly report to the Regional Controller or the authorized officer and the State Government.",
            "The report should cover operations from January to June and July to December each year.",
            "Exploration licensees must also submit a geological report within three months of completing operations, identifying areas suitable for a mining lease."
        ]
    },
    "restriction_on_disclosure": {
        "patterns": ["What restrictions are there on disclosing information?", "Tell me about disclosure restrictions", "Explain information disclosure rules"],
        "responses": [
            "The holder of an exploration license is restricted from disclosing information, schemes, and reports to anyone other than the specified government or authorities without prior approval from the Central Government."
        ]
    },
    "mining_plan_review": {
        "patterns": ["When is a mining plan review required?", "Tell me about the review of mining plans", "Explain the mining plan review process"],
        "responses": [
            "Mining or mineral processing operations may require a mining plan review if discontinued for a period exceeding specified days.",
            "The holder should submit a notice to the authorized officer and the State Government under rule 28.",
            "The exact requirements depend on the specific rules mentioned in the respective sections."
        ]
    },
    "schedule_amendments": {
        "patterns": ["What amendments are there in Schedule I?", "Tell me about changes in Schedule I", "Explain the amendments to forms"],
        "responses": [
            "Amendments in Schedule I include inserting 'or exploration license' wherever 'composite license' is mentioned.",
            "Forms such as Form-A, Form-B, Form-H, Form-I, Form-J, Form-K, and Form-N now include 'or exploration license'.",
            "Important instructions for filling the form have also been updated in Form-B."
        ]
    },
 "violation_penalty": {
        "patterns": ["What are the penalties for rule violations?", "Tell me about penalties under Rule 45", "Explain the fines for non-compliance"],
        "responses": [
            "Under Rule 45, the amount to be paid in case of violation depends on the nature of the violation.",
            "For non-submission or incomplete/wrong/false information in monthly returns (Form F1, F2, F3), the penalty ranges from ₹5,000 to ₹10,000 per day, depending on the leased area and production capacity.",
            "Similar penalties apply for violations in annual returns (Form G1, G2, G3), monthly returns (Form L), and annual returns (Form M).",
            "Specific rules (e.g., Rule 11, Rule 12, Rule 18, etc.) also have associated fines for contravention.",
            "Feel free to ask about a specific rule or type of violation for more details."
        ]
    },
    "specific_rule_penalty": {
        "patterns": ["What is the penalty for violating Rule 11?", "Tell me about fines for Rule 18 violations", "Explain penalties under Rule 28"],
        "responses": [
            "Certainly! Here are some examples of fines for specific rules:",
            "- Rule 11 (Mining operations under mining lease): ₹1,000 per day, subject to a maximum of ₹5,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes.",
            "- Rule 18 (Beneficiation studies to be carried out): ₹1,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes, ₹5,00,000 for other cases.",
            "- Rule 28 (Notice of temporary discontinuance of work in mines and obligations of lease holders): ₹1,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes, ₹5,00,000 for other cases."
        ]
    },
 "mineral_rules_amendment": {
        "patterns": ["Tell me about the Minerals (Evidence of Mineral Contents) Amendment Rules, 2024", "Explain the recent amendments in mineral rules", "What changes have been made in the Minerals (Evidence of Mineral Contents) Rules, 2015?"],
        "responses": [
            "The Minerals (Evidence of Mineral Contents) Amendment Rules, 2024 have been introduced with some key changes.",
            "One significant change is in Rule 2, where additional criteria have been specified for minerals with a grade equal to or more than the threshold value under the Atomic Minerals Concession Rules, 2016.",
            "Rule 5 has been updated to include 'section 11D' in addition to 'section 11'.",
            "Rule 7 now has provisions specific to minerals in Part D of the First Schedule to the Act. Proposals for such minerals are to be submitted to the Central Government, and a committee will assess mineral potentiality for blocks proposed by individuals.",
            "Schedule III includes information on where proposals should be submitted - either to the State Government or the Central Government."
        ]
    },
    "submission_procedures": {
        "patterns": ["How do I submit a proposal under the Minerals (Evidence of Mineral Contents) Rules?", "Tell me about the submission procedures for mineral proposals"],
        "responses": [
            "To submit a proposal under the Minerals (Evidence of Mineral Contents) Rules:",
            "- Follow the guidelines in Schedule III of the rules.",
            "- Depending on your case, submit your proposal to either the State Government or the Central Government.",
            "- Ensure that you comply with the relevant sections mentioned in the amendments, such as section 11 or section 11D.",
            "- For minerals in Part D of the First Schedule, proposals need to be submitted to the Central Government and evaluated by a committee.",
            "Feel free to ask if you have specific questions about the submission process."
        ]
    },
"mineral_rules": {
        "patterns": ["What are the Mineral Conservation and Development Rules?", "Tell me about the Minerals (Evidence of Mineral Contents) Rules, 2015.", "Explain the amendments in the Minerals (Evidence of Mineral Contents) Amendment Rules, 2024.", "When do the amended rules come into force?", "Who approves proposals for minerals specified in Part D of the First Schedule to the Act?"],
        "responses": ["The Mineral Conservation and Development Rules govern mining activities.", "The Minerals (Evidence of Mineral Contents) Rules, 2015, specify requirements for mineral exploration.", "The amendments in 2024 introduce changes to how mineral potentiality is identified and submitted.", "The amended rules came into force on January 21, 2024.", "Proposals for minerals in Part D are approved by the Central Government."]
    },
    "amendment_details": {
        "patterns": ["What changes were made in Rule 2?", "Explain the amendments in Rule 5.", "Tell me about the committee mentioned in Rule 7(1B).", "What destinations are mentioned in Rule 3 of Schedule III?", "When were the Principal Rules last amended?"],
        "responses": ["Rule 2 was amended to include specifications for minerals based on grade.", "Rule 5 was amended to include references to section 11D.", "Rule 7(1B) introduces a committee to identify mineral potentiality.", "Rule 3 in Schedule III clarifies submission destinations: State Government or Central Government.", "The Principal Rules were last amended on December 14, 2021."]
    },
    "conclusion": {
        "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
        "responses": ["The amendments focus on specifying mineral criteria, submitting proposals to the Central Government, and establishing a committee for mineral potentiality.", "The conclusion highlights changes related to minerals, submission destinations, and the effective date of the amendments."]
    },
 "atomic_minerals_rules": {
        "patterns": ["What are the Atomic Minerals Concession Rules, 2016?", "Tell me about the amendments in the Atomic Minerals Concession (Amendment) Rules, 2023.", "When did the amended rules come into force?", "What is the penalty for contravening the specified rules?"],
        "responses": ["The Atomic Minerals Concession Rules, 2016, regulate the mining of atomic minerals.", "The Amendment Rules of 2023 introduce changes in the penalty provisions and deposit exploration details.", "The amended rules came into force on September 22, 2023.", "The penalty for contravening specified rules may include imprisonment up to two years, a fine up to five lakhs, or both."]
    },
    "amendment_details": {
        "patterns": ["What changes were made in Rule 37?", "Explain the substituted entries in Schedule B, Part III.", "Tell me about the exploration details for Rare metal and REE deposits in pegmatites, reefs, and veins/pipes (Serial Number III)."],
        "responses": ["Rule 37 introduces new provisions for penalties related to specified rules.", "In Schedule B, Part III, Serial Number III for Rare metal and REE deposits in pegmatites, scout drilling/pitting/trenching is required at 10 to 25 pits/trenches per sq.km.", "Exploration details for Rare metal and REE deposits in pegmatites include scout drilling/pitting/trenching and exploratory open pit or boreholes at specified intervals."]
    },
    "conclusion": {
        "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
        "responses": ["The amendments focus on introducing penalties for contraventions, updating deposit exploration details, and specifying the rules related to penalties.", "The conclusion highlights changes in penalty provisions, exploration requirements, and the effective date of the amendments."]
    },
 "mineral_auction_rules": {
        "patterns": ["What are the Mineral (Auction) Rules, 2015?", "Explain the amendments in the Mineral (Auction) Amendment Rules, 2023.", "When did the amended rules come into force?", "What changes were made in rule 5, sub-rule (2)?", "Tell me about the new rules 9B and 17B."],
        "responses": [
            "The Mineral (Auction) Rules, 2015, govern the auctioning of minerals.",
            "The Amendment Rules of 2023 introduce changes in rule 5, sub-rule (2) and add rules 9B and 17B.",
            "The amended rules came into force on September 1, 2023.",
            "Rule 5, sub-rule (2) now includes a proviso allowing the use of land details from the Prime Minister Gati Shakti - National Master Plan or state land record portals for land classification.",
            "Rule 9B relates to the conduct of auction of mining leases by the Central Government, and Rule 17B relates to the conduct of auction of composite licenses by the Central Government."
        ]
    },
    "rule_details": {
        "patterns": ["Explain rule 9B.", "What are the key points of rule 17B?", "Tell me about the provisions of conducting an auction by the Central Government under section 11D."],
        "responses": [
            "Rule 9B outlines the procedure for conducting auctions of mining leases by the Central Government, including the intimation of details by the State Government, receipt of geological reports, termination of leases, and the role of the Central Government.",
            "Rule 17B details the procedure for the Central Government to conduct auctions of composite licenses, involving the intimation of details by the State Government, receipt of geological reports, termination of licenses, and the role of the Central Government.",
            "For conducting an auction under section 11D, the Central Government follows rules 5 to 9 (for mining leases) or rules 16 and 17 (for composite licenses) and informs the State Government of the preferred bidder upon successful completion."
        ]
    },
    "conclusion": {
        "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
        "responses": [
            "The amendments focus on incorporating land details from the Gati Shakti platform, updating rules related to land classification, and introducing procedures for the Central Government to conduct auctions of mining leases and composite licenses."
        ]
    },
"minerals_concession_rules": {
        "patterns": ["What are the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2016?", "When did these rules come into force?", "What minerals do these rules apply to?", "Define 'railway' and 'run-of-mine'."],
        "responses": [
            "The Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2016, were established under the Mines and Minerals (Development and Regulation) Act, 1957.",
            "These rules came into force on the date of their publication in the Official Gazette.",
            "These rules apply to all minerals, except minor minerals defined under Section 3(e) and minerals listed in Part A and Part B of the First Schedule to the Act.",
            "In these rules, 'railway' and 'run-of-mine' are defined as per the Indian Railways Act and refer to raw, unprocessed material obtained from the mineralized zone of a lease area, respectively."
        ]
    },
    "definitions": {
        "patterns": ["Define 'illegal mining,' 'mineral concession,' 'run-of-mine,' etc.", "Explain the meaning of 'value of estimated resources.'"],
        "responses": [
            "'Illegal mining' refers to unauthorized reconnaissance, prospecting, or mining operations in an area without the required mineral concession.",
            "'Mineral concession' includes reconnaissance permit, non-exclusive reconnaissance permit, prospecting licence, prospecting licence-cum-mining lease, or a mining lease.",
            "'Run-of-mine' is the raw material obtained after blasting or digging from the mineralized zone of a lease area.",
            "'Value of estimated resources' is the product of the estimated quantity of mineral resources granted by a concession and the average price per metric tonne of the mineral, as published by the Indian Bureau of Mines for the relevant state, over the preceding twelve months."
        ]
    },
    "rights_of_existing_holders": {
        "patterns": ["What are the rights of the existing holders of mineral concessions?", "Explain the procedure for a holder of a reconnaissance permit to apply for a prospecting licence."],
        "responses": [
            "Existing holders of mineral concessions have specific rights under these rules. For example, a holder of a reconnaissance permit may apply for a prospecting licence under certain conditions.",
            "For a holder of a reconnaissance permit, the rules outline a procedure for applying for a prospecting licence, including the application format, acknowledgement process, fees, and the role of the State and Central Governments."
        ]
    },
    "renewal_of_prospecting_licence": {
        "patterns": ["How can one renew a prospecting licence?", "What information needs to be provided for the renewal of a prospecting licence?"],
        "responses": [
            "The renewal of a prospecting licence involves submitting an application ninety days before expiry, providing a statement with reasons, a report on prospecting operations, expenditure details, and justification for additional time.",
            "The State Government acknowledges the renewal application, and the process includes a non-refundable fee, possible condonation of delay, and timely disposal by the State Government."
        ]
    },
 "prospecting_to_mining": {
        "patterns": [
            "How can I apply for a mining lease?",
            "What are the steps to obtain a mining lease from a prospecting licence?",
            "Tell me about mining lease application process.",
            "What is required for mining lease after a prospecting licence?"
        ],
        "responses": [
            "To obtain a mining lease from a prospecting licence, you need to follow these steps:",
            "1. Submit an application for a mining lease within three months after the prospecting licence expiry.",
            "2. The State Government will acknowledge your application and may require a non-refundable fee of Rs. 5 lakhs per sq. km.",
            "3. Fulfill conditions specified in sub-clause (i) to sub-clause (iv) of clause (b) of sub-section (2) of Section 10A.",
            "4. The State Government decides within 60 days, may forward to Central Government for approval.",
            "5. Once approved, obtain necessary consents, approvals, permits, provide performance security, and sign an agreement with the State Government.",
            "6. The State Government executes a mining lease deed within 90 days. If not executed, the order may be revoked."
        ]
    },
    "section_100_c": {
        "patterns": [
            "What are the rights under Section 100(c)?",
            "Tell me about the rights for mining lease grant under Section 100(c).",
            "Explain Section 100(c) of the Mines and Minerals Act.",
            "What happens if conditions in the letter of intent or previous approval are not fulfilled?"
        ],
        "responses": [
            "Under Section 100(c) of the Mines and Minerals Act:",
            "1. The applicant submits a letter of compliance for the conditions mentioned in the letter of intent or previous approval.",
            "2. The State Government issues an order for grant of the mining lease within 60 days, subject to condition verification.",
            "3. If conditions are not fulfilled, the State Government may refuse to grant a mining lease.",
            "4. Upon issuance of an order, the applicant must furnish a performance security to the State Government and sign an agreement.",
            "5. The mining lease should be executed and registered on or before January 11, 2017, else the right is forfeited."
        ]
    },
 "prospecting_lease_through_auction": {
        "patterns": [
            "Tell me about the composite license granted through auction.",
            "What is the format for prospecting license deed under the Mineral (Auction) Rules, 2015?",
            "Explain the mining lease deed for successful bidders under Mineral (Auction) Rules."
        ],
        "responses": [
            "1. The prospecting license deed for the composite license is in the format specified in Schedule V.",
            "2. Mining lease deed for successful bidders under Mineral (Auction) Rules, 2015, is in the format specified in Schedule VII."
        ]
    },
    "renewal_of_prospecting_license": {
        "patterns": [
            "How can I renew a prospecting license?",
            "Tell me about the renewal process for a composite license prospecting stage.",
            "What is the fee for renewing a prospecting license?"
        ],
        "responses": [
            "To renew a prospecting license:",
            "1. Apply at least ninety days before the expiry with reasons, details of operations, expenditure, man-days, and justification for additional time.",
            "2. State Government acknowledges the renewal application within three days.",
            "3. Pay a non-refundable fee of Rs. 1000 per sq. km.",
            "4. State Government may condone delay if applied before prospecting license stage expiry.",
            "5. State Government decides on renewal before the prospecting license expires."
        ]
    },
    "terms_and_conditions_of_licenses": {
        "patterns": [
            "What are the terms and conditions of a prospecting license?",
            "Explain the conditions for licensees under the Mines and Minerals Act.",
            "Tell me about the licensee's responsibilities and obligations."
        ],
        "responses": [
            "1. Licensee may win minerals within specified limits without payment or on payment of royalty.",
            "2. Licensee can carry away minerals for chemical, metallurgical, ore-dressing, and test purposes with written permission.",
            "3. Licensee convicted of illegal mining may have the license canceled and performance security forfeited.",
            "4. Licensee must report the discovery of any mineral within sixty days.",
            "5. Licensee needs to comply with Act and rules, restore affected land, maintain accurate accounts, and follow specific conditions."
        ]
    },
 "mining_lease": {
        "patterns": [
            "What are the conditions for a mining lease?",
            "Tell me about mining lease terms",
            "Explain the obligations in a mining lease",
            "What are the restrictions on mining operations?",
            "How should a lessee report accidents in a mining lease?"
        ],
        "responses": [
            "In a mining lease, the lessee must pay yearly dead rent, commence operations within two years, and follow restrictions on mining activities.",
            "Mining lease terms involve payment obligations, environmental responsibilities, and government rights.",
            "Obligations include payment of dead rent, surface rent, and water rate, along with maintaining accurate records and restoring affected landforms.",
            "Restrictions include no mining within 50 meters from railways without permission and not interfering with public grounds or village roads.",
            "Accidents in mining operations must be promptly reported to the Deputy Commissioner or Collector."
        ]
    },
 "mining_lease": {
        "patterns": [
            "What are the conditions for a mining lease?",
            "Tell me about mining lease terms",
            "Explain the obligations in a mining lease",
            "What are the restrictions on mining operations?",
            "How should a lessee report accidents in a mining lease?"
        ],
        "responses": [
            "In a mining lease, the lessee must pay yearly dead rent, commence operations within two years, and follow restrictions on mining activities.",
            "Mining lease terms involve payment obligations, environmental responsibilities, and government rights.",
            "Obligations include payment of dead rent, surface rent, and water rate, along with maintaining accurate records and restoring affected landforms.",
            "Restrictions include no mining within 50 meters from railways without permission and not interfering with public grounds or village roads.",
            "Accidents in mining operations must be promptly reported to the Deputy Commissioner or Collector."
        ]
    },
    "prospecting_license": {
        "patterns": [
            "What is the process for renewal of a prospecting license?",
            "Tell me about prospecting license conditions",
            "What are the rights and responsibilities of a prospecting license holder?",
            "Explain the restrictions on prospecting operations.",
            "How does force majeure affect prospecting license terms?"
        ],
        "responses": [
            "Prospecting license renewal requires an application with reasons, reports of prospecting operations, and justifications for additional time.",
            "Conditions include the right to win minerals for testing, restoration of land, and compliance with Act and rules.",
            "Rights involve winning and carrying minerals, restoration of landforms, and maintaining accurate accounts.",
            "Restrictions include obtaining permission for clearing land, complying with rules, and reporting mineral discoveries.",
            "Force majeure events may extend the prospecting license period, and delay due to force majeure is not considered a breach."
        ]
    },
  "mining_lease_info": {
        "patterns": ["What are the conditions of a mining lease?", "Tell me about mining lease rules", "Explain mining lease provisions", "What rights do mining lease holders have?", "How does pre-emption work in mining leases?", "What are the conditions for lease termination?", "Tell me about discovered minerals in a mining lease"],
        "responses": [
            "Mining lease conditions include payment of rents and royalties, surface rent, and water rate.",
            "Mining lease holders have rights for mining operations, including working the mines, constructing infrastructure, obtaining materials, and more.",
            "Pre-emption in mining leases allows the government to purchase discovered minerals from the lease holder.",
            "Lease termination can occur due to default, illegal mining, or failure to comply with conditions.",
            "Mining lease holders must follow specific conditions outlined by the State Government.",
            "Discovered minerals in a mining lease can be disposed of after inclusion in the lease deed."
        ]
    },
 "mining_lease_conditions": {
        "patterns": ["What are the conditions of a mining lease?", "Tell me about mining lease rules", "Explain mining lease provisions", "What are the obligations of a mining lease holder?"],
        "responses": [
            "Mining lease conditions include payment of rents, royalties, surface rent, and water rate.",
            "Lessee must commence mining operations within two years, conduct operations properly, and restore the landform after mining operations.",
            "Mining lease holders should give preference to tribals and those displaced by mining operations.",
            "Conditions also cover areas like pre-emption rights, storage of unutilized ores, employment preferences, and more."
        ]
    },
    "mining_lease_discovered_minerals": {
        "patterns": ["What happens if a new mineral is discovered?", "Tell me about discovered minerals in a mining lease", "Can a mining lease holder dispose of discovered minerals?"],
        "responses": [
            "Holder of a mining lease through auction may win and dispose of the mineral discovered after inclusion in the mining lease deed.",
            "In case of a mining lease not granted through auction, the state government may exercise pre-emption rights and pay the holder the cost of production for the mineral.",
            "Discovery of minerals not specified in the lease by the holder not granted through auction doesn't grant disposal rights; the state government may exercise pre-emption."
        ]
    },
    "mining_lease_further_conditions": {
        "patterns": ["What other conditions can be in a mining lease?", "Tell me about additional mining lease conditions", "What conditions can the state government impose for mineral development?"],
        "responses": [
            "Mining leases may contain conditions on payment modes, compensation for land damage, tree felling restrictions, surface operations, reporting accidents, indemnity to the government, and more.",
            "State governments may impose further conditions for mineral development in the interests of the industry."
        ]
    },
 "mining_plan": {
        "patterns": ["What is a mining plan?", "How to get a mining plan approved?", "Tell me about the components of a mining plan", "Who approves the mining plan?", "Explain the process for mining plan modification", "What qualifications are required for preparing a mining plan?"],
        "responses": [
            "A mining plan is required for any mining operations and must be approved by the Indian Bureau of Mines or the State Government.",
            "To get a mining plan approved, submit it to the designated officer along with the required fee. The Indian Bureau of Mines will then review the application.",
            "A mining plan includes details like the lease hold area, geology, exploration program, mining methods, environmental management, and more.",
            "The mining plan is approved by the Indian Bureau of Mines or the State Government as per the established system.",
            "For modification, submit the proposed changes to the approving authority. The process follows the rules outlined in the mining plan regulations.",
            "To prepare a mining plan, one needs a degree in mining engineering or a post-graduate degree in geology, along with five years of supervisory experience in mining."
        ]
    },
 "mining_plan": {
        "patterns": ["What is a mining plan?", "How to get a mining plan approved?", "Tell me about the components of a mining plan", "Who approves the mining plan?", "Explain the process for mining plan modification", "What qualifications are required for preparing a mining plan?"],
        "responses": [
            "A mining plan is required for any mining operations and must be approved by the Indian Bureau of Mines or the State Government.",
            "To get a mining plan approved, submit it to the designated officer along with the required fee. The Indian Bureau of Mines will then review the application.",
            "A mining plan includes details like the lease hold area, geology, exploration program, mining methods, environmental management, and more.",
            "The mining plan is approved by the Indian Bureau of Mines or the State Government as per the established system.",
            "For modification, submit the proposed changes to the approving authority. The process follows the rules outlined in the mining plan regulations.",
            "To prepare a mining plan, one needs a degree in mining engineering or a post-graduate degree in geology, along with five years of supervisory experience in mining."
        ]
    },
    "mining_operations": {
        "patterns": ["What are the details of mining operations?", "Tell me about the methods of mining operation", "Explain environmental management in mining", "What is a progressive mine closure plan?"],
        "responses": [
            "Mining operations involve activities like excavation, drilling, blasting, waste handling, mineral beneficiation, and more.",
            "Methods of mining operation include excavation techniques, drilling and blasting processes, mineral handling, and beneficiation.",
            "Environmental management in mining includes baseline information, impact assessment, and mitigation measures.",
            "A progressive mine closure plan outlines the steps for the closure of the mine in a phased manner as defined in the rules."
        ]
    },
 "lease_conditions": {
        "patterns": ["Tell me about the conditions in a mining lease", "What are the restrictions for felling trees in a leased area?", "Explain compensation for land damage in a mining lease", "How are accidents reported in mining operations?"],
        "responses": [
            "Conditions in a mining lease cover aspects like rent and royalties payment, compensation for land damage, restrictions on tree felling, and more.",
            "Restrictions on felling trees in a leased area are outlined in the lease conditions imposed by the State Government.",
            "Compensation for land damage is a part of the mining lease conditions and is specified based on the rules set by the State Government.",
            "Accidents in mining operations are required to be reported as per the conditions outlined in the mining lease. The reporting process is crucial for safety measures."
        ]
    },
 "mining_lease_expiry": {
        "patterns": ["What happens on the expiry of a mining lease?", "Tell me about mining lease expiry", "Expiry of mining lease"],
        "responses": ["After the expiry of a mining lease, it will be put up for auction following specified procedures."]
    },
    "right_of_first_refusal": {
        "patterns": ["Explain the right of first refusal in mining leases", "What is the right of first refusal?", "How does right of first refusal work?"],
        "responses": ["The holder of a mining lease for captive purposes has the right of first refusal at the time of auction after lease expiry."]
    },
    "lapsing_of_mining_lease": {
        "patterns": ["What is the process of lapsing of a mining lease?", "When does a mining lease lapse?", "Lapsing of mining lease"],
        "responses": ["A mining lease may lapse if mining operations don't commence within two years or are discontinued for two continuous years."]
    },
    "surrender_of_mining_lease": {
        "patterns": ["How can a mining lease be surrendered?", "Tell me about surrendering a mining lease", "Surrender process of mining lease"],
        "responses": ["The lessee may apply for the surrender of the entire area of the mining lease after giving a notice of not less than twelve calendar months from the intended date of surrender."]
    },
    "termination_of_mining_lease": {
        "patterns": ["Under what circumstances can a mining lease be terminated?", "Tell me about mining lease termination", "Mining lease termination conditions"],
        "responses": ["The State Government can terminate a mining lease if there's a breach or if the lessee transfers the lease without following proper procedures."]
    },
    "transfer_of_mining_lease": {
        "patterns": ["Explain the process of transferring a mining lease", "How can a mining lease be transferred?", "Mining lease transfer rules"],
        "responses": ["Transfer of a mining lease is allowed with the previous approval of the State Government."]
    },
 "prospecting_license": {
        "patterns": ["How to obtain a prospecting license?", "Tell me about prospecting license", "Procedure for obtaining a prospecting license", "Prospecting license application"],
        "responses": ["To obtain a prospecting license, follow the procedure in Chapter IX. Provide specific details for a tailored response."]
    },
    "mining_lease": {
        "patterns": ["What are the conditions for a mining lease?", "Tell me about mining lease", "Mining lease application process", "Conditions of mining lease"],
        "responses": ["Conditions for a mining lease are detailed in Rule 29. For more information, ask a specific question."]
    },
    "transfer_of_license": {
        "patterns": ["Can I transfer a prospecting license?", "Procedure for transferring mining lease", "Transfer of license rules"],
        "responses": ["Yes, you can transfer a prospecting license. The process is in Rule 23. Specify your question for more details."]
    },
    "working_of_mines": {
        "patterns": ["Prohibition of working of mines", "Can I work on my mines?", "State Government's role in mine working"],
        "responses": ["The State Government may prohibit mining if there's a contravention. See Rule 32 for details."]
    },
    "returns_statements": {
        "patterns": ["What returns and statements are required?", "Filing returns for mining lease", "Statement submission for prospecting license"],
        "responses": ["Holders must provide returns as specified in Rule 33. Specify for more detailed information."]
    },
    "penalty": {
        "patterns": ["Penalty for contravention of mining rules", "Consequences of violating mining regulations", "What happens if rules are not followed?"],
        "responses": ["Violations lead to penalties under Rule 34. For specifics, specify your question."]
    },
"revision_application": {
        "patterns": [
            "How to apply for revision?",
            "Revision application process",
            "Central Government revision application",
            "Applying for order revision"
        ],
        "responses": [
            "To apply for revision, submit an application to the Central Government within three months of the order. See Rule 35 for details."
        ]
    },
    "application_fee": {
        "patterns": [
            "What is the application fee for revision?",
            "Revision application fee",
            "Central Government application fee",
            "How much to pay for revision application?"
        ],
        "responses": [
            "The application fee for revision is a bank draft of rupees ten thousand. Details are in Rule 35(2)."
        ]
    },
    "application_timeframe": {
        "patterns": [
            "How long do I have to apply for revision?",
            "Revision application timeframe",
            "Deadline for revision application",
            "When to submit a revision application?"
        ],
        "responses": [
            "You have three months from the date of order communication to apply for revision. Refer to Rule 35(1) for more information."
        ]
    },
    "impleaded_parties": {
        "patterns": [
            "Who are impleaded parties in a revision application?",
            "Implication of parties in revision",
            "Parties involved in a revision application",
            "Revision application participants"
        ],
        "responses": [
            "In every application under Rule 35(1), parties to whom a mineral concession was granted for the same area shall be impleaded."
        ]
    },
    "comments_submission": {
        "patterns": [
            "How to submit comments for a revision application?",
            "Providing feedback on revision application",
            "Comments on Rule 35 application",
            "Submitting feedback on revision order"
        ],
        "responses": [
            "Comments on a revision application must be submitted within three months from the date of communication. See Rule 35(5) for details."
        ]
    },
    "order_decision": {
        "patterns": [
            "How does the Central Government decide on a revision application?",
            "Central Government's role in revision orders",
            "Decision-making in revision applications",
            "Outcome of a revision application"
        ],
        "responses": [
            "The Central Government may confirm, modify, or set aside the order after considering comments. Rule 35(4) provides details."
        ]
    },
    "stay_execution": {
        "patterns": [
            "Can the execution of an order be stayed during a revision application?",
            "Order execution stay in revision application",
            "Stay of execution in Rule 35 revision",
            "Central Government stay during revision"
        ],
        "responses": [
            "Yes, the Central Government may stay the execution of the order for sufficient cause during the revision process. See Rule 35(5) for more information."
        ]
    },
 "associated_minerals": {
        "patterns": [
            "Which minerals are associated?",
            "List of associated minerals",
            "Group of minerals in Section 6",
            "Tell me about associated minerals"
        ],
        "responses": [
            "The associated minerals for the purposes of Section 6 are categorized as follows:\n(a) Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, Lepidolite, Pitchblende, Samarskite, Scheelite, Topaz, Tantalite, Tourmaline.\n(b) Iron, Manganese, Titanium, Vanadium, and Nickel minerals.\n(c) Lead, Zinc, Copper, Cadmium, Arsenic, Antimony, Bismuth, Cobalt, Nickel, Molybdenum, Uranium minerals, Gold, Silver, Arsenopyrite, Chalcopyrite, Pyrite, Pyrrhotite, and Pentlandite.\n(d) Chromium, Osmiridium, Platinum, and Nickel minerals.\n(e) Kyanite, Sillimanite, Corundum, Dumortierite, and Topaz.\n(f) Gold, Silver, Tellurium, Selenium, and Pyrite.\n(g) Fluorite, Chalcocite, Selenium, and minerals of Zinc, Lead, and Silver.\n(h) Tin and Tungsten minerals.\n(i) Limestone and Magnesite.\n(j) Ilmenite, Monazite, Zircon, Rutile, Leucoxene, Garnet, and Sillimanite.\n(k) Sulphides of Copper and Iron.\n(l) Magnetite and Apatite.\n(m) Magnesite and Chromite.\n(n) Celestite and Phosphatic Nodules.",
            "These associated minerals play a significant role in various geological and industrial processes. If you have specific questions about any of these minerals or their applications, feel free to ask!"
        ]
    },
 "minerals_valuation": {
        "patterns": [
            "What is sale value?",
            "Explain sale value",
            "Tell me about mineral valuation",
            "How is royalty calculated?"
        ],
        "responses": [
            "Sale value is the gross amount payable by the purchaser as indicated in the sale invoice in arms-length transactions. It excludes taxes and no deduction is made for royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust.",
            "Royalty is chargeable on processed minerals within the leased area or unprocessed run-of-mine removed to a processing plant outside the leased area. The Act specifies royalty calculation based on ad valorem or London Metal Exchange prices, and in some cases, tonnage basis.",
            "Provisional assessment and adjustment allow lessees to calculate royalty, payments to District Mineral Foundation, and payments to National Mineral Exploration Trust based on the latest available average sale price. Adjustments are made after the publication of average sale prices by the Indian Bureau of Mines.",
            "The royalty chargeable on a dry basis for metallic ores is based on the metal contained in the ore, calculated using London Metal Exchange or London Bullion Market Association prices. The average sale price of mineral grade/concentrate is computed using the ex-mine price, and the Indian Bureau of Mines publishes this data monthly."
        ]
    },
  "payments": {
        "patterns": [
            "How are fees and deposits made?",
            "Explain payment procedure",
            "Tell me about making payments",
            "What is the payment process?"
        ],
        "responses": [
            "Any amount payable under the Act or rules, except for the revision petition fee under rule 35(2), should be paid as specified by the State Government.",
            "The State Government may charge simple interest at the rate of 24% per annum on rent, royalty, or fee, excluding the fee under rule 35(2), from the sixtieth day after the expiry of the payment due date until the payment is made.",
            "Mining lease or prospecting licence-cum-mining lease holders are required to pay monies to the District Mineral Foundation and the National Mineral Exploration Trust as per Section 9B and Section 9C, respectively. Additionally, payments under Rule 13 of the Mineral (Auction) Rules, 2015, involve paying the applicable amount quoted under Rule 8 on a monthly basis."
        ]
    },
 "minerals_valuation": {
        "patterns": [
            "What is sale value?",
            "Explain sale value",
            "Tell me about the sale invoice",
            "How is sale value calculated?"
        ],
        "responses": [
            "Sale value is the gross amount payable by the purchaser as indicated in the sale invoice, excluding taxes, if any. No deduction is made from the gross amount for royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust when computing sale value.",
            "Royalty is charged based on whether processing of run-of-mine occurs within or outside the leased area. The Act specifies different methods for calculating royalty based on an ad valorem basis or London Metal Exchange or London Bullion Market Association prices.",
            "Provisional assessment and adjustment of royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust are made at the time of removal or consumption of mineral from the mining lease area."
        ]
    },
    "mining_rules_revision": {
        "patterns": [
            "How can one apply for revision?",
            "Explain the revision process",
            "Tell me about rule 35",
            "What is the procedure for revision?"
        ],
        "responses": [
            "Any person aggrieved by an order made by the State Government or other authority can apply for revision within three months of the date of communication of the order. The application should be accompanied by a bank draft or bank transfer for the application fee.",
            "On receipt of the application for revision, copies are sent to the State Government or other authority and all impleaded parties. Comments and counter-comments are sought, and the Central Government may confirm, modify, or set aside the order.",
            "The Controller General of Indian Bureau of Mines has the power to issue necessary directions to give effect to the provisions of this chapter."
        ]
    },
    "associated_minerals": {
        "patterns": [
            "Tell me about associated minerals",
            "What are the groups of associated minerals?",
            "Explain Section 6",
            "Which minerals are associated?"
        ],
        "responses": [
            "Associated minerals are grouped for the purposes of Section 6. The groups include various minerals such as Apatite, Beryl, Cassiterite, Columbite, and more.",
            "The associated minerals are categorized into groups such as Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, and more.",
            "The groups of associated minerals are identified for the purposes of Section 6, and they include various minerals like Iron, Manganese, Titanium, Vanadium, Nickel, Lead, Zinc, Copper, and many more.",
            "For the purposes of Section 6, the associated minerals are classified into groups such as Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, Lepidolite, and more."
        ]
    },
    "minerals_chapter": {
        "patterns": [
            "Tell me about minerals in a specific chapter",
            "Explain a particular chapter on minerals",
            "Details about a mining chapter",
            "Which chapter covers certain minerals?"
        ],
        "responses": [
            "Chapter IX covers the procedure for obtaining a prospecting license or mining lease for lands where minerals vest exclusively in a person other than the Government.",
            "Chapter X pertains to revision and provides details on the application process, order considerations, and comments from involved parties.",
            "Chapter XI focuses on associated minerals, outlining the groups, classifications, and names of minerals falling under Section 6.",
            "Chapter XII delves into minerals valuation, discussing topics like sale value, payment of royalty, provisional assessment, and the computation of average sale prices."
        ]
    },
  "compensation": {
        "patterns": [
            "Tell me about compensation payment",
            "How is compensation determined?",
            "Explain compensation for damage",
            "When is compensation payable?"
        ],
        "responses": [
            "The holder of a mineral concession is liable to pay annual compensation to the occupier of the surface land. The amount is determined by an officer appointed by the State Government and is based on the average annual net income for agricultural land or the average annual letting value for non-agricultural land for the previous three years.",
            "For agricultural land, annual compensation is based on the average annual net income from the cultivation of similar land for the previous three years. For non-agricultural land, it is based on the average annual letting value of similar land for the previous three years. The compensation must be paid on or before the specified date by the State Government.",
            "After the cessation of mining activities due to the expiry, lapsing, surrender, or termination of a mineral concession, the State Government assesses the damage to the land caused by reconnaissance, prospecting, or mining operations. The compensation amount is determined by an officer appointed by the State Government within one year from the date of cessation of mining activities.",
            "The annual compensation, as mentioned in sub-rule (1), is payable on or before the date specified by the State Government."
        ]
    },
    "penalty": {
        "patterns": [
            "What is the penalty for rule contravention?",
            "Explain the penalty under these rules",
            "Tell me about rule violation consequences",
            "What happens if someone breaks the rules?"
        ],
        "responses": [
            "Any contravention of these rules is punishable with imprisonment for up to two years or a fine up to rupees five lakhs, or both. In the case of a continuing contravention, there is an additional fine of up to rupees fifty thousand for every day after conviction for the first contravention.",
            "Violating these rules can lead to imprisonment for a maximum of two years or a fine up to rupees five lakhs, or both. For continuing violations, there is an extra fine of up to rupees fifty thousand for each day after the first conviction.",
            "Breaking these rules may result in imprisonment for up to two years or a fine up to rupees five lakhs, or both. If the contravention continues, an additional fine of up to rupees fifty thousand is imposed for each day after the first conviction.",
            "Consequences of breaking these rules include imprisonment for a term not exceeding two years or a fine up to rupees five lakhs, or both. For continuous violations, there is an additional fine of up to rupees fifty thousand for each day after the initial conviction."
        ]
    },
  "repeal_and_saving": {
        "patterns": [
            "Explain repeal and saving under Chapter XVI",
            "Tell me about cessation of Mineral Concession Rules, 1960",
            "What happens on the commencement of these rules?",
            "How are references to Mineral Concession Rules, 1960 replaced?"
        ],
        "responses": [
            "On the commencement of these rules, the Mineral Concession Rules, 1960, cease to be in force with respect to minerals for which the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, are applicable. This ceasing is applicable to things done or omitted to be done before such commencement.",
            "With respect to minerals covered by these rules, any reference to the Mineral Concession Rules, 1960, in rules made under the Act or any other document is deemed to be replaced with Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, to the extent not repugnant to the context.",
            "The Mineral Concession Rules, 1960, cease to have effect for minerals under the jurisdiction of the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, upon the commencement of these rules.",
            "References to the Mineral Concession Rules, 1960, in any rules made under the Act or other documents are considered as replaced by Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, to the extent not conflicting with the context upon the commencement of these rules."
        ]
    },
    "amalgamation_of_leases": {
        "patterns": [
            "Explain amalgamation of leases under Chapter XVII",
            "How can two or more leases be amalgamated?",
            "Tell me about the conditions for amalgamation",
            "What happens to the period of amalgamated leases?"
        ],
        "responses": [
            "In the interest of mineral development and with reasons recorded in writing, the State Government may permit the amalgamation of two or more adjoining leases held by a lessee.",
            "The State Government has the authority to permit the amalgamation of two or more adjoining leases held by a lessee if it is in the interest of mineral development. The lessee must provide reasons for the amalgamation, recorded in writing.",
            "For amalgamation of leases, the period of amalgamated leases is co-terminus with the lease whose period will expire first. The State Government permits amalgamation based on reasons recorded in writing and the interest of mineral development.",
            "Two or more adjoining leases held by a lessee may be amalgamated by the State Government if it is in the interest of mineral development. The period of amalgamated leases aligns with the lease whose period expires first."
        ]
    },
    "extent_of_area_granted": {
        "patterns": [
            "What is the extent of the area granted under a mineral concession?",
            "Explain the scope of the area granted",
            "Tell me about the non-mineralised area under a mineral concession",
            "How is the area defined under a mineral concession?"
        ],
        "responses": [
            "The extent of the area granted under a mineral concession includes the non-mineralised area required for all activities falling under the definition of a mine as defined in clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
            "Under a mineral concession, the extent of the granted area encompasses the non-mineralised area necessary for activities falling under the definition of a mine according to clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
            "The extent of the area granted under a mineral concession includes the non-mineralised area needed for all activities defined as a mine under clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
            "The area granted under a mineral concession extends to the non-mineralised area required for activities falling within the definition of a mine as per clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952)."
        ]
    },
    "rectify_apparent_mistakes": {
        "patterns": [
            "How can apparent mistakes be rectified?",
            "Tell me about rectification of mistakes",
            "Explain the correction of errors in orders",
            "What is the procedure for rectification of mistakes?"
        ],
        "responses": [
            "Any clerical or arithmetical mistake in any order passed by the Government or any other authority or officer under these rules, and any error arising due to accidental slip or omission, may be corrected by the Government, authority, or officer within two years from the date of the order.",
            "Rectification of any clerical or arithmetical mistake in an order passed by the Government or any other authority or officer under these rules, and correction of errors due to accidental slip or omission, can be done by the Government, authority, or officer within two years from the date of the order.",
            "Corrections of clerical or arithmetical mistakes in orders passed by the Government or any other authority or officer under these rules and rectification of errors arising due to accidental slip or omission can be carried out by the Government, authority, or officer within two years from the date of the order.",
            "Within two years from the date of the order, the Government, authority, or officer may rectify any clerical or arithmetical mistake in an order or correct errors arising due to accidental slip or omission under these rules."
        ]
    },
    "copies_of_licences_and_leases": {
        "patterns": [
            "What information must be supplied to the Government?",
            "Explain the supply of copies of licences and leases",
            "Tell me about the annual return to be supplied",
            "How is the information on mineral concessions provided to the Government?"
        ],
        "responses": [
            "A copy of every mineral concession granted or renewed under the Act and rules made thereunder shall be supplied by each State Government within two months of such grant or renewal to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety.",
            "Every State Government must supply a copy of each mineral concession granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety within two months of the grant or renewal.",
            "Each State Government is required to supply a consolidated annual return of all mineral concessions granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, not later than the 30th day of June following the year to which the return relates. A copy of this return is also supplied to the Director General, Directorate General of Mines Safety.",
            "The Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety, must be supplied with a copy of every mineral concession granted or renewed under the Act and rules within two months of such grant or renewal."
        ]
    },
    "copies_of_licences_and_leases": {
        "patterns": [
            "What information must be supplied to the Government?",
            "Explain the supply of copies of licences and leases",
            "Tell me about the annual return to be supplied",
            "How is the information on mineral concessions provided to the Government?"
        ],
        "responses": [
            "A copy of every mineral concession granted or renewed under the Act and rules made thereunder shall be supplied by each State Government within two months of such grant or renewal to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety.",
            "Every State Government must supply a copy of each mineral concession granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety within two months of the grant or renewal.",
            "Each State Government is required to supply a consolidated annual return of all mineral concessions granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, not later than the 30th day of June following the year to which the return relates. A copy of this return is also supplied to the Director General, Directorate General of Mines Safety.",
            "The Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety, must be supplied with a copy of every mineral concession granted or renewed under the Act and rules within two months of such grant or renewal."
        ]
    }
}



training_data = []
labels = []


for intent, data in intents.items():
    if 'patterns' in data:
        for pattern in data['patterns']:
            training_data.append(pattern.lower())
            labels.append(intent)


# Placeholder for your custom model and vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english", max_df=0.8, min_df=1)
X_train = vectorizer.fit_transform(training_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, labels, test_size=0.4, random_state=42)

model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train, Y_train)

# Function to predict intent
def predict_intent(user_input):
    user_input = user_input.lower()
    input_vector = vectorizer.transform({user_input})
    intent = model.predict(input_vector)[0]
    return intent

# Function to get response based on intent
def intentAns(query):
    user_input = str(query)
    intent = predict_intent(user_input)
    try:
        if intent in intents and 'responses' in intents[intent]:
            responses = intents[intent]['responses']
            if responses and len(responses) > 0:
                response = random.choice(responses)
                if len(response)>10:
                    return response  # Return the response instead of printing
                else:
                    client = OpenAI(api_key= "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")

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                            "role": "user",
                            "content": f"{query}"
                            }
                        ],
                        temperature=0.7,
                        max_tokens=128,
                        top_p=0.9,
                        frequency_penalty=0,
                        presence_penalty=0
                        )

                    ans = response.choices[0].message.content
                    return ans
    except Exception as e:
        client = OpenAI(api_key= "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")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": f"{query}"
                }
            ],
            temperature=0.7,
            max_tokens=128,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
            )

        ans = response.choices[0].message.content
        return ans

# Your routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    response = get_Chat_response(input_text)
    return jsonify({"response": response})  # Return response as JSON

def get_Chat_response(text):
    response = intentAns(text)
    return response

if __name__ == '__main__':
    app.run()








# from flask import Flask, render_template, request, jsonify
# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# import nltk
# import openai

# app = Flask(__name__)




# intents = {
#     "mining_amendments": {
#         "patterns": ["What amendments were introduced by G.S.R. 737(E)?",
#                      "Can you explain the modifications made by G.S.R. 737(E)?",
#                      "Tell me about the amendments in the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
#                      "What changes were made to the mining regulations by G.S.R. 737(E)?",
#                      "Explain the modifications introduced by G.S.R. 737(E) in the mining rules."],
#         "responses": ["G.S.R. 737(E) introduced amendments to the Mines and Minerals (Development and Regulation) Act, 1957.",
#                       "The Central Government, under section 13 of the Mines and Minerals (Development and Regulation) Act, 1957, made modifications to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
#                       "Modifications were made to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023 by G.S.R. 737(E).",
#                       "G.S.R. 737(E) brought changes to the mining regulations specified in the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023.",
#                       "The amendments introduced by G.S.R. 737(E) pertain to the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession (Amendment) Rules, 2023."]
#     },
#     "rule_44_modification": {
#         "patterns": ["What modification was made to Rule 44 by G.S.R. 737(E)?",
#                      "Can you explain clause (ia) of Rule 44 as per G.S.R. 737(E)?",
#                      "Tell me about the changes introduced to Rule 44 by G.S.R. 737(E)."],
#         "responses": ["G.S.R. 737(E) introduced clause (ia) to Rule 44 regarding Lithium.",
#                       "Rule 44 was modified by G.S.R. 737(E) to include clause (ia) concerning Lithium.",
#                       "The modification made to Rule 44 by G.S.R. 737(E) involves the addition of clause (ia) related to Lithium."]
#     },
#     "rule_45_modification": {
#         "patterns": ["Explain the amendments to Rule 45 by G.S.R. 737(E).",
#                      "What changes were made to sub-rules 5 and 6 of Rule 45 according to G.S.R. 737(E)?",
#                      "Can you provide details about the modifications made to Rule 45 by G.S.R. 737(E)?"],
#         "responses": ["G.S.R. 737(E) introduced modifications to Rule 45, including sub-rules 5 and 6.",
#                       "Rule 45 was amended by G.S.R. 737(E) with changes to sub-rules 5 and 6.",
#                       "The amendments made to Rule 45 by G.S.R. 737(E) include modifications to sub-rules 5 and 6."]
#     },
#     "publication_history": {
#         "patterns": ["What is the publication history of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016?",
#                      "Can you provide details about the publication of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016?",
#                      "Tell me about the Gazette publication of the Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016."],
#         "responses": ["The Minerals (Other than Atomic and Hydro Carbons Energy Mineral) Concession Rules, 2016 were published in the Gazette of India on March 4, 2016, under number G.S.R. 279(E)."]
#     },
#     "mining_amendments_736": {
#         "patterns": ["What amendments were introduced by G.S.R. 736(E)?",
#                      "Can you explain the modifications made by G.S.R. 736(E)?",
#                      "Tell me about the amendments in the Mines and Minerals (Development and Regulation) Act, 1957 introduced by G.S.R. 736(E).",
#                      "What changes were made to the mining regulations by G.S.R. 736(E)?",
#                      "Explain the modifications introduced by G.S.R. 736(E) in the mining laws."],
#         "responses": ["G.S.R. 736(E) introduced amendments to the Mines and Minerals (Development and Regulation) Act, 1957.",
#                       "The Central Government, under sub-section (3) of section 9 of the Mines and Minerals (Development and Regulation) Act, 1957, made modifications to the mining regulations.",
#                       "Modifications were made to the Mines and Minerals (Development and Regulation) Act, 1957 by G.S.R. 736(E).",
#                       "G.S.R. 736(E) brought changes to the mining regulations specified in the Mines and Minerals (Development and Regulation) Act, 1957.",
#                       "The amendments introduced by G.S.R. 736(E) pertain to the Mines and Minerals (Development and Regulation) Act, 1957."]
#     },
#     "item_28A_amendment": {
#         "patterns": ["What amendment was made for Lithium by G.S.R. 736(E)?",
#                      "Can you explain the modification introduced for Lithium by G.S.R. 736(E)?",
#                      "Tell me about the change in charge for Lithium as per G.S.R. 736(E)."],
#         "responses": ["G.S.R. 736(E) introduced a charge of three per cent of London Metal Exchange price on the Lithium metal in the ore produced."]
#     },
#     "item_33_amendment": {
#         "patterns": ["What change was made to Monazite by G.S.R. 736(E)?",
#                      "Can you explain the modification introduced for Monazite by G.S.R. 736(E)?",
#                      "Tell me about the amendment regarding Monazite as per G.S.R. 736(E)."],
#         "responses": ["G.S.R. 736(E) changed 'Monazite' to 'Monazite occurring in beach sand minerals'."]
#     },
#     "item_34A_amendment": {
#         "patterns": ["Explain the amendments made for Niobium by G.S.R. 736(E).",
#                      "What changes were introduced for Niobium by G.S.R. 736(E)?",
#                      "Can you provide details about the modifications for Niobium by G.S.R. 736(E)?"],
#         "responses": ["G.S.R. 736(E) introduced amendments for Niobium, specifying charges for both primary and by-product Niobium."]
#     },
#     "item_38A_amendment": {
#         "patterns": ["What amendment was made for Rare Earth Elements by G.S.R. 736(E)?",
#                      "Can you explain the modification introduced for Rare Earth Elements by G.S.R. 736(E)?",
#                      "Tell me about the change in charge for Rare Earth Elements as per G.S.R. 736(E)."],
#         "responses": ["G.S.R. 736(E) introduced a charge of one per cent of average sale price of Rare Earth Oxide on the Rare Earth Oxide contained in the ore produced."]
#     },
#     "offshore_mining_act_commencement": {
#         "patterns": ["When does the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 come into force?",
#                      "Can you tell me the commencement date of the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023?",
#                      "When will the Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 be effective?"],
#         "responses": ["The Offshore Areas Mineral (Development and Regulation) Amendment Act, 2023 comes into force on the 17th of August, 2023."]
#     },
#     "mines_and_minerals_act_commencement": {
#         "patterns": ["When does the Mines and Minerals (Development and Regulation) Amendment Act, 2023 come into force?",
#                      "Can you tell me the commencement date of the Mines and Minerals (Development and Regulation) Amendment Act, 2023?",
#                      "When will the Mines and Minerals (Development and Regulation) Amendment Act, 2023 be effective?"],
#         "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023 comes into force on the 17th of August, 2023."]
#     },
#     "signatory": {
#         "patterns": ["Who signed S.O. 3684(E)?",
#                      "Can you provide the name of the signatory for S.O. 3684(E)?",
#                      "Tell me who signed the document S.O. 3684(E)."],
#         "responses": ["S.O. 3684(E) was signed by Dr. Veena Kumari Dermal, Joint Secretary."]
#     },
#     "act_details": {
#         "patterns": ["Mines and Minerals (Development and Regulation) Amendment Act, 2023", "details about Mines and Minerals (Development and Regulation) Amendment Act, 2023", "Mines and Minerals Act 2023", "MMDR Amendment Act 2023"],
#         "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023, received the assent of the President on August 9, 2023. It aims to further amend the Mines and Minerals (Development and Regulation) Act, 1957."]
#     },
#     "act_amendments": {
#         "patterns": ["What are the amendments in the Mines and Minerals Act?", "Amendments in Mines and Minerals Act 2023", "Changes in Mines and Minerals Act", "MMDR Act amendments"],
#         "responses": ["The Mines and Minerals (Development and Regulation) Amendment Act, 2023, introduced several amendments to the Mines and Minerals (Development and Regulation) Act, 1957. These include amendments related to exploration licenses, termination of prospecting licenses and mining leases, restrictions on mineral concessions, maximum area for mineral concessions, procedure for obtaining mineral concession, application for mineral concession, grant of exploration licenses through auction, and more."]
#     },
#     "offshore_areas_mineral_act": {
#         "patterns": ["What is the Offshore Areas Mineral (Development and Regulation) Act, 2002?", "Can you provide information about the Offshore Areas Mineral Act?", "What does the Offshore Areas Mineral Act govern?"],
#         "responses": ["The Offshore Areas Mineral (Development and Regulation) Act, 2002, governs the development and regulation of mineral resources in India's offshore areas, including territorial waters, continental shelf, exclusive economic zone, and other maritime zones."]
#     },
#     "short_title_and_commencement": {
#         "patterns": ["What is the short title of the Offshore Areas Mineral Act?", "When did the Offshore Areas Mineral Act come into force?"],
#         "responses": ["The short title of the Offshore Areas Mineral Act is the Offshore Areas Mineral (Development and Regulation) Act, 2002. It came into force on the date specified by the Central Government through a notification in the Official Gazette."]
#     },
#     "expediency_of_union_control": {
#         "patterns": ["Why was it declared expedient for the Union to control regulation of mines and mineral development in offshore areas?"],
#         "responses": ["It was declared expedient in the public interest for the Union to control the regulation of mines and mineral development in offshore areas to the extent provided in the act."]
#     },
#     "application": {
#         "patterns": ["To which areas does the Offshore Areas Mineral Act apply?", "What is the scope of the Offshore Areas Mineral Act?"],
#         "responses": ["The Offshore Areas Mineral Act applies to all minerals in offshore areas, excluding mineral oils and hydrocarbons related thereto."]
#     },
#     "population": {
#         "patterns": ["What is the population of the United States?", "Population of USA", "How many people live in the USA?", "What's the current population of the USA?"],
#         "responses": ["The population of the United States is approximately 331 million people as of 2022."]
#     },
#     "capital": {
#         "patterns": ["What is the capital of France?", "Capital of France", "Where is the capital of France?", "France's capital city"],
#         "responses": ["The capital of France is Paris."]
#     },
#     "area": {
#         "patterns": ["What is the total area of Canada?", "Area of Canada", "How big is Canada?", "Canada's total land area"],
#         "responses": ["Canada has a total area of approximately 9.98 million square kilometers."]
#     },
#     "highest_mountain": {
#         "patterns": ["What is the tallest mountain in the world?", "Highest mountain on Earth", "Name the highest peak on the planet", "World's tallest mountain"],
#         "responses": ["The tallest mountain in the world is Mount Everest, with a height of approximately 8,848.86 meters."]
#     },
#     "deepest_ocean": {
#         "patterns": ["What is the deepest ocean on Earth?", "Deepest ocean in the world", "Name the deepest part of the ocean", "Which ocean has the greatest depth?"],
#         "responses": ["The deepest ocean in the world is the Pacific Ocean, with the Mariana Trench being the deepest part, reaching depths of approximately 10,994 meters."]
#     },
#     "largest_desert": {
#         "patterns": ["What is the largest desert on Earth?", "Biggest desert in the world", "Name the largest arid region on the planet", "Which desert is the largest?"],
#         "responses": ["The largest desert in the world is the Sahara Desert, covering an area of approximately 9.2 million square kilometers."]
#     },
#     "power_of_entry": {
#         "patterns": [
#             "What are the powers of entry, inspection, search, and seizure?",
#             "Can officers enter and inspect mines?",
#             "What can authorized officers do regarding mines?",
#             "Explain the authority to search and seize in mines."
#         ],
#         "responses": [
#             "Under the Act, authorized officers have various powers:",
#             "- They can enter and inspect mines at reasonable times.",
#             "- They can weigh, draw samples, and take measurements of mineral stocks.",
#             "- They're authorized to survey, take samples, and measurements in mines.",
#             "- Officers can examine documents, books, registers, and records related to mines.",
#             "- They have the authority to order the production of documents.",
#             "- Officers can also examine individuals connected with the mines."
#         ]
#     },
#     "search_and_seizure": {
#         "patterns": [
#             "What are the procedures for search and seizure?",
#             "Can officers search mines without a warrant?",
#             "Explain the seizure of vessels and mines.",
#             "How can officers enforce search and seizure?"
#         ],
#         "responses": [
#             "Regarding search and seizure:",
#             "- Officers can search mines without a warrant to ascertain compliance with the Act.",
#             "- They can stop or board vessels engaged in regulated activities.",
#             "- Officers can seize vessels, mines, equipment, and minerals involved in violations.",
#             "- They're empowered to arrest individuals committing violations."
#         ]
#     },
#     "offences": {
#         "patterns": [
#             "What are the offences under the Act?",
#             "What penalties apply for violations?",
#             "Explain the penalties for obstructing officers.",
#             "Can companies be held liable for offences?"
#         ],
#         "responses": [
#             "The Act specifies various offences and penalties:",
#             "- Undertaking operations without necessary permits or licences.",
#             "- Failing to provide required data or obstructing authorized officers.",
#             "- Penalties include fines, imprisonment, and confiscation of vessels and minerals.",
#             "- Companies and individuals can be held liable for violations."
#         ]
#     },
#     "civil_liability": {
#         "patterns": [
#             "How is civil liability determined?",
#             "Explain the liability for contravening terms and conditions.",
#             "Who has jurisdiction over civil liability cases?",
#             "What powers do authorized officers have for civil liability?"
#         ],
#         "responses": [
#             "Civil liability is determined as follows:",
#             "- Contravention of general and particular terms and conditions incurs liability.",
#             "- Only authorized officers designated by the Central Government have jurisdiction.",
#             "- Officers can file applications against permittees, licensees, or lessees for civil wrongs.",
#             "- Authorized officers have powers similar to civil courts for adjudication."
#         ]
#     },
#     "extension_of_enactments": {
#         "patterns": [
#             "Can enactments be extended to offshore areas?",
#             "How are Indian enactments applied to offshore areas?",
#             "Explain the extension of laws to offshore areas."
#         ],
#         "responses": [
#             "Enactments can be extended to offshore areas as follows:",
#             "- The Central Government can extend existing laws with restrictions and modifications.",
#             "- Provisions are made for enforcement of such laws in offshore areas.",
#             "- Extended enactments have the same effect as if the area were part of India."
#         ]
#     },
#     "compounding_of_offences": {
#         "patterns": [
#             "Is there provision for compounding offences?",
#             "Can offences under the Act be compounded?",
#             "Explain how offences can be compounded."
#         ],
#         "responses": [
#             "Offences under the Act can be compounded as follows:",
#             "- Administering authorities or authorized officers can compound offences.",
#             "- Offenders can pay a specified sum, not exceeding the maximum fine, to compound the offence.",
#             "- Compounding an offence prevents further legal proceedings against the offender."
#         ]
#     },
#     "recovery_of_sums": {
#         "patterns": [
#             "How are sums due to the Central Government recovered?",
#             "Explain the recovery of licence fees and royalties.",
#             "Can overdue amounts be recovered as arrears of land revenue?"
#         ],
#         "responses": [
#             "Sums due to the Central Government are recovered as follows:",
#             "- Licence fees, royalties, and other sums can be recovered as arrears of land revenue.",
#             "- The administering authority can issue certificates for recovery.",
#             "- Recovered sums, along with interest, are given priority over other assets."
#         ]
#     },
#     "delegation_of_powers": {
#         "patterns": [
#             "Is there provision for delegation of powers?",
#             "Can the Central Government delegate its powers?",
#             "Explain how powers are delegated under the Act."
#         ],
#         "responses": [
#             "Powers under the Act can be delegated as follows:",
#             "- The Central Government can delegate its powers to subordinate officers or authorities.",
#             "- Delegation is subject to specified conditions and matters.",
#             "- Delegated officers or authorities can exercise powers in relation to designated matters."
#         ]
#     },
#     "protection_of_action": {
#         "patterns": [
#             "Are there provisions for protecting actions taken in good faith?",
#             "Is there protection against legal proceedings for actions under the Act?",
#             "Explain the protection provided for good faith actions."
#         ],
#         "responses": [
#             "Actions taken in good faith are protected as follows:",
#             "- No legal proceedings can be initiated against individuals for actions done in good faith.",
#             "- Protection extends to actions under the Act and its rules.",
#             "- This provision safeguards individuals acting in accordance with the Act."
#         ]
#     },
#     "mineral_rates": {
#         "patterns": ["royalty rates", "fixed rent rates", "mineral rates", "rates of royalty", "rates of fixed rent", "mining rates"],
#         "responses": [
#             "The rates of royalty and fixed rent are provided below:\n\nRoyalty Rates:\n1. Brown ilmenite (leucoxene), Ilmenite, Rutile and Zircon: 2% of sale price on ad valorem basis.\n2. Dolomite: ₹40 per tonne.\n3. Garnet: 3% of sale price on ad valorem basis.\n4. Gold: 1.5% of London Bullion Market Association price.\n5. Limestone and Lime mud: ₹40 per tonne.\n6. Manganese Ore: 3% of sale price on ad valorem basis.\n7. Monazite: ₹125 per tonne.\n8. Sillimanite: 2.5% of sale price on ad valorem basis.\n9. Silver: 5% of London Metal Exchange price.\n10. All other minerals: 10% of sale price on ad valorem basis.",
#             "Fixed Rent Rates:\nRates are provided in rupees per standard block per annum, varying based on the year of the lease.\n1st Year: [rate]\n2nd to 5th Year: [rate]\n6th to 10th Year: [rate]\n11th Year onwards: [rate]"
#         ]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "S.O. 575(E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
#         "responses": ["The notification S.O. 575(E) was issued by the Ministry of Mines, Government of India, on February 3rd, 2023.", "The purpose of the notification is to confer powers to the Jharkhand Exploration and Mining Corporation Limited, Ranchi, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is the Jharkhand Exploration and Mining Corporation Limited, Ranchi.", "The notification imposes a condition that the corporation must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "S.O. 208(E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
#         "responses": ["The notification S.O. 208(E) was issued by the Ministry of Mines, Government of India, on January 12th, 2023.", "The purpose of the notification is to confer powers to M/s. FCI Aravali Gypsum and Minerals India Limited, a Central Government Company, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is M/s. FCI Aravali Gypsum and Minerals India Limited.", "The notification imposes a condition that the company must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 425 (E)", "notification date", "purpose of notification", "authority notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 425 (E) was issued by the Ministry of Mines, Government of India, on June 22nd, 2021.", "The purpose of the notification is to confer powers to Hutti Gold Mines Company Limited, Karnataka, under the Mines and Minerals (Development and Regulation) Act, 1957.", "The authority notified by the notification is Hutti Gold Mines Company Limited.", "The notification imposes a condition that the company must share prospecting operation data with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 864(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 864(E) was issued by the Ministry of Mines, Government of India, on November 20th, 2019.", "The purpose of the notification is to notify certain entities under the Mines and Minerals (Development and Regulation) Act, 1957.", "The entities notified are Tamil Nadu Minerals Limited, Tamil Nadu Magnesite Limited, Tamil Nadu Cements Corporation Limited (all State Government Undertakings), and NLC India Limited (a Central Government Undertaking under the administrative control of the Ministry of Coal).", "The notification imposes a condition that the data generated by the prospecting operations must be shared with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 707(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 707(E) was issued by the Ministry of Mines, Government of India, on July 27th, 2018.", "The purpose of the notification is to notify Hindustan Copper Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "Hindustan Copper Limited is the entity notified in this notification.", "The notification imposes a condition that Hindustan Copper Limited must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 389(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 389(E) was issued by the Ministry of Mines, Government of India, on April 23rd, 2018.", "The purpose of the notification is to notify Odisha Mineral Exploration Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "Odisha Mineral Exploration Corporation Limited is the entity notified in this notification.", "The notification imposes a condition that Odisha Mineral Exploration Corporation Limited must share the data generated by prospecting operations with the State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 40(E)", "notification date", "purpose of notification", "entity notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 40(E) was issued by the Ministry of Mines, Government of India, on January 18th, 2018.", "The purpose of the notification is to notify National Thermal Power Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "National Thermal Power Corporation Limited is the entity notified in this notification.", "The notification imposes a condition that National Thermal Power Corporation Limited must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "notification_details": {
#         "patterns": ["notification details", "G.S.R. 1325(E)", "notification date", "purpose of notification", "entities notified", "condition of notification"],
#         "responses": ["The notification G.S.R. 1325(E) was issued by the Ministry of Mines, Government of India, on October 24th, 2017.", "The purpose of the notification is to notify M/s. Odisha Mining Corporation Limited and M/s. West Bengal Mineral Development and Trading Corporation Limited under the Mines and Minerals (Development and Regulation) Act, 1957.", "M/s. Odisha Mining Corporation Limited and M/s. West Bengal Mineral Development and Trading Corporation Limited are the entities notified in this notification.", "The notification imposes a condition that the notified entities must share the data generated by prospecting operations with the concerned State Government.", "The notification came into force upon its publication in the Official Gazette."]
#     },
#     "mining_notifications": {
#       "patterns": [
#         "mining notification",
#         "Mines and Minerals Act notification",
#         "Ministry of Mines notification",
#         "MMDR Act notification"
#       ],
#       "responses": [
#         "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
#       ]
#     },
#     "notification_details": {
#       "patterns": [
#         "details about notification",
#         "details of notification",
#         "notification details",
#         "more info about notification",
#         "what does the notification say"
#       ],
#       "responses": [
#         "Of course! Please provide the notification number or the date of the notification you want details about."
#       ]
#     },
#     "mining_notifications": {
#       "patterns": [
#         "mining notification",
#         "Mines and Minerals Act notification",
#         "Ministry of Mines notification",
#         "MMDR Act notification"
#       ],
#       "responses": [
#         "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
#       ]
#     },
#     "notification_details": {
#       "patterns": [
#         "details about notification",
#         "details of notification",
#         "notification details",
#         "more info about notification",
#         "what does the notification say"
#       ],
#       "responses": [
#         "Of course! Please provide the notification number or the date of the notification you want details about."
#       ]
#     },
#     "notification_2015_07_06": {
#       "patterns": [
#         "notification dated 6th July 2015",
#         "Mines and Minerals Act notification G.S.R. 538(E)"
#       ],
#       "responses": [
#         "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
#       ]
#     },
#     "mining_notifications": {
#       "patterns": [
#         "mining notification",
#         "Mines and Minerals Act notification",
#         "Ministry of Mines notification",
#         "MMDR Act notification"
#       ],
#       "responses": [
#         "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
#       ]
#     },
#     "notification_details": {
#       "patterns": [
#         "details about notification",
#         "details of notification",
#         "notification details",
#         "more info about notification",
#         "what does the notification say"
#       ],
#       "responses": [
#         "Of course! Please provide the notification number or the date of the notification you want details about."
#       ]
#     },
#     "notification_2015_07_06": {
#       "patterns": [
#         "notification dated 6th July 2015",
#         "Mines and Minerals Act notification G.S.R. 538(E)"
#       ],
#       "responses": [
#         "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
#       ]
#     },
#     "mining_notifications": {
#       "patterns": [
#         "mining notification",
#         "Mines and Minerals Act notification",
#         "Ministry of Mines notification",
#         "MMDR Act notification"
#       ],
#       "responses": [
#         "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
#       ]
#     },
#     "notification_details": {
#       "patterns": [
#         "details about notification",
#         "details of notification",
#         "notification details",
#         "more info about notification",
#         "what does the notification say"
#       ],
#       "responses": [
#         "Of course! Please provide the notification number or the date of the notification you want details about."
#       ]
#     },
#     "notification_2015_07_06": {
#       "patterns": [
#         "notification dated 6th July 2015",
#         "Mines and Minerals Act notification G.S.R. 538(E)"
#       ],
#       "responses": [
#         "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
#       ]
#     },
#     "mining_notifications": {
#       "patterns": [
#         "mining notification",
#         "Mines and Minerals Act notification",
#         "Ministry of Mines notification",
#         "MMDR Act notification"
#       ],
#       "responses": [
#         "Sure, I can provide information about mining notifications. Please specify the notification you're interested in."
#       ]
#     },
#     "notification_details": {
#       "patterns": [
#         "details about notification",
#         "details of notification",
#         "notification details",
#         "more info about notification",
#         "what does the notification say"
#       ],
#       "responses": [
#         "Of course! Please provide the notification number or the date of the notification you want details about."
#       ]
#     },
#     "notification_2015_07_06": {
#       "patterns": [
#         "notification dated 6th July 2015",
#         "Mines and Minerals Act notification G.S.R. 538(E)"
#       ],
#       "responses": [
#         "The notification dated 6th July 2015, G.S.R. 538(E), notifies several entities for the purposes of the second proviso to sub-section (1) of section 4 of the Mines and Minerals (Development and Regulation) Act, 1957."
#       ]
#     },
#     "PMKKKY_Objective": {
#         "patterns": ["What is the objective of PMKKKY?", "What are the objectives of Pradhan Mantri Khanij Kshetra Kalyan Yojana?", "What does PMKKKY aim to achieve?"],
#         "responses": ["The objective of PMKKKY is to implement developmental and welfare projects/programs in mining affected areas, mitigate adverse impacts of mining on environment and socio-economics, and ensure sustainable livelihoods for affected people."]
#     },
#     "DMF_Establishment": {
#         "patterns": ["When was the District Mineral Foundation established?", "What is District Mineral Foundation?", "When was DMF established?"],
#         "responses": ["The District Mineral Foundation (DMF) was established in all districts affected by mining operations through the amendment of the Mines & Minerals (Development & Regulation) Act, 1957, in 2015."]
#     },
#     "DMF_Funding": {
#         "patterns": ["How is the District Mineral Foundation funded?", "What are the sources of funding for DMF?", "Who funds the DMFs?"],
#         "responses": ["DMFs are funded by statutory contributions from mining lease holders."]
#     },
#     "PMKKKY_Implementation": {
#         "patterns": ["How is PMKKKY implemented?", "Who implements Pradhan Mantri Khanij Kshetra Kalyan Yojana?"],
#         "responses": ["PMKKKY is implemented by the District Mineral Foundations (DMFs) of the respective districts using the funds accruing to the DMF."]
#     },
#     "PMKKKY_Composition": {
#         "patterns": ["What is the composition of PMKKKY?", "Who is included in the composition of PMKKKY?"],
#         "responses": ["The composition of PMKKKY includes MPs, MLAs, and MLCs in the Governing Council, as per the directive of the Central Government."]
#     },
#     "PMKKKY_Directions": {
#         "patterns": ["What directions were issued regarding PMKKKY?", "What directions did the Central Government issue regarding PMKKKY?"],
#         "responses": ["The Central Government issued directions regarding composition and utilization of funds by DMFs, including the inclusion of MPs, MLAs, and MLCs in the Governing Council, prohibition of fund transfers to state exchequer, and preparation of a five-year Perspective Plan."]
#     },
#     "PMKKKY_Revision": {
#         "patterns": ["Are there any revised PMKKKY guidelines?", "What are the revised PMKKKY guidelines?"],
#         "responses": ["Yes, revised PMKKKY guidelines have been issued under section 9B (3) of the MMDR Act 1957 after consultation with stakeholders."]
#     },
#     "Affected_Areas_People": {
#         "patterns": ["How are affected areas and people identified under PMKKKY?", "What is the process of identifying affected areas and people under PMKKKY?"],
#         "responses": ["Affected areas and people are identified based on direct and indirect impacts of mining operations, including displacement, economic dependence, and environmental consequences."]
#     },
#     "Utilization_of_Funds": {
#         "patterns": ["How are the funds under PMKKKY utilized?", "What is the utilization of funds under PMKKKY?"],
#         "responses": ["PMKKKY funds are utilized for high priority sectors such as drinking water supply, healthcare, education, livelihood generation, and environment preservation, with a minimum allocation of 70% to directly affected areas."]
#     },
#     "composition_and_functions": {
#         "patterns": "The Chairman of Governing Council and Managing Committee shall be the District Magistrate/Deputy Commissioner/Collector of the district. No other person shall function as Chairman of the Governing Council and/or Managing Committee.",
#         "responses": {
#             "mp": "MPs representing mining-affected areas shall be members of the Governing Council. Each Lok Sabha MP with mining-affected areas in their constituency shall be a member. If an MP's constituency spans multiple districts, they shall be a member of the Governing Council in each district. Similarly, a Rajya Sabha MP shall be a member of the Governing Council of one district.",
#             "mla": "MLAs representing mining-affected areas shall be members of the Governing Council. If an MLA's constituency spans multiple districts, they shall be a member of the Governing Council in each district.",
#             "mlc": "Members of Legislative Council shall be a member of the Governing Council of one district chosen by them.",
#             "meetings": "The Governing Council shall convene at least twice a year, with meeting dates scheduled according to the convenience of MP members.",
#             "managing_committee": "The Managing Committee shall consist of the District Magistrate/Deputy Commissioner/Collector as Chairman and senior district officers responsible for project execution. Elected representatives or nominated non-official members are not part of the committee.",
#             "meeting_frequency": "The Managing Committee shall meet at least once every quarter."
#         }
#     },
#     "introduction": {
#         "patterns": ["what is the introduction?", "introduction", "purpose of the committee", "committee's purpose"],
#         "response": "The committee was formed to examine the issue of misclassification of grades of iron ore and other minerals and to suggest measures for preventing it. It was also tasked with exploring the adoption of advanced technology in this regard."
#     },
#     "methodology": {
#         "patterns": ["what methodology was adopted?", "methodology adopted by the committee", "how did the committee conduct its study?"],
#         "response": "The committee adopted a consultative approach, including presentations by the Indian Bureau of Mines (IBM) and state government representatives. Written comments were invited from industry associations, companies, and other stakeholders. A sub-committee visited iron ore mines in Odisha to understand sampling issues and explore new technologies. Thirteen meetings were held to discuss the findings and recommendations."
#     },
#     "sampling_dispatch_transportation": {
#         "patterns": ["what is the existing procedure of sampling, dispatch, and transportation?", "existing procedure of sampling", "existing procedure of dispatch", "existing procedure of transportation"],
#         "response": "State governments have established rules to prevent illegal mining, transportation, and storage of minerals, including measures such as check-posts and weigh-bridges. After excavation, minerals are sorted into different sizes and grades and dispatched to end-users based on demand. Online systems are used for royalty assessment and payment, with weighment done at mine sites. Penalties are prescribed for transporting minerals without lawful authority."
#     },
#     "average_sale_price": {
#         "patterns": ["what is the average sale price (ASP)?", "relevance of ASP", "how is ASP calculated?", "ASP calculation"],
#         "response": "ASP is crucial for calculating revenue to state governments, particularly in lease auctions. It is used for royalty assessment, valuation of mineral blocks for auctions, and calculating bid premiums. Online returns are filed by mining lease holders, and ASP compilation is based on these returns."
#     },
#     "iron_ore_misclassification": {
#         "patterns": ["impact of misclassification", "effect of misreporting", "revenue loss due to misclassification", "state government concerns", "iron ore grade classification", "measures to prevent misclassification", "technological solutions for misclassification"],
#         "responses": ["Misclassification of iron ore grades can have significant financial implications, particularly for state governments and other stakeholders involved in the mining and mineral industry. It can lead to revenue losses due to differences in ASP and impact revenue generation through royalty, auction premiums, etc. State governments are concerned about mine owners replacing higher grades of iron ore with lower grades in their reports, which directly affects revenue generation. Measures to address misclassification include implementing robust mechanisms for sampling and grade declaration to prevent revenue losses. Some states have adopted IT-based systems for sampling and analysis. Technological solutions such as handheld XRF analyzers and cryptography for data verification during transportation are also being considered."],
#     },
#     "subcommittee_constitution": {
#         "patterns": ["constitution of sub-committee", "formation of sub-committee", "sub-committee visit to Odisha"],
#         "responses": ["To understand the issues involved in sampling and declaration of grades of iron ore and explore new technologies, a sub-committee was formed. The members visited Odisha from 31.05.2022 to 02.06.2022."],
#     },
#     "subcommittee_members": {
#         "patterns": ["members of the sub-committee", "who participated in the tour"],
#         "responses": ["The members of the sub-committee were: 1) Sh. Dheeraj Kumar, Deputy Secretary (Mines) 2) Sh. S.K. Adhikari, CMG, IBM 3) Sh. Sanjay Khare, Dy. Director, Govt. of Chhattisgarh (nominated by the State Government) 4) Sh. Salil Behera, Jt. Director of Mines, Govt. of Odisha (nominated by the State Government) 5) Sh. Sambhav Jain, Sr. Manager(Legal), NALCO Ltd./ Ministry of Mines"],
#     },
#     "visit_details": {
#         "patterns": ["details of the visit", "purpose of the visit", "observations from the visit"],
#         "responses": ["The sub-committee visited Odisha to understand sampling and grade declaration issues. They visited Joda East Iron Ore Mine (exempted) and Jajang Iron Ore Mine (non-exempted). They noted shortcomings in the stacking, sampling, and analysis process."],
#     },
#     "sampling_technology": {
#         "patterns": ["presentation by technology suppliers", "latest sampling technology", "auto-sampling and sample analysis"],
#         "responses": ["The committee explored technologies for iron ore grading. They considered options like cross-belt conveyor sampling system, auger system, and laser analyzer. The laser technology was found promising to resolve misclassification issues."],
#     },
#     "mineral_recommendations": {
#         "patterns": ["recommendations", "mineral recommendations", "what are the recommendations"],
#         "response": "The recommendations include implementing an IT-enabled system for sampling, analysis, and transportation monitoring, integration of leaseholder systems with government monitoring, incentives for adopting new technologies, use of IT-based grade information systems, automated sampling and analysis, videography of sampling process, random sampling, and regular audits."
#     },
#     "state_government_powers": {
#         "patterns": ["state government powers", "powers of state governments", "what are the powers of state governments"],
#         "response": "State governments have complete legislative and administrative powers related to transportation and storage of minerals. They also receive revenue from mineral production in the form of royalty, auction premium, and other payments."
#     },
#     "penal_action": {
#         "patterns": ["penal action", "what happens in case of illegal transportation", "what are the penalties for illegal transportation"],
#         "response": "State governments have sufficient powers to take penal action against illegal transportation of minerals. Any mineral transported in contravention of state government rules shall be considered 'without lawful authority' and attract penalties as prescribed under the MMDR Act."
#     },
#     "recommendations_implementation": {
#         "patterns": ["implementation of recommendations", "how to implement recommendations", "what is the process to implement recommendations"],
#         "response": "Recommendations should be implemented by state governments through rules under section 23C of the MMDR Act and other guidelines. The central government may issue necessary advice to facilitate implementation and maintain uniformity in rules across states."
#     },
#     "misclassification_of_minerals": {
#         "patterns": ["misclassification of minerals", "how to prevent misclassification", "recommendations for preventing misclassification"],
#         "response": "To prevent misclassification of minerals, it is recommended to implement an IT-enabled system with minimal human intervention, use of technologies like continuous online analyzers for sampling and analysis, videography of sampling process, random sampling, and regular audits."
#     },
#     "transportation_recommendations": {
#         "patterns": ["transportation recommendations", "how to improve mineral transportation", "recommendations for mineral transportation"],
#         "response": "Recommendations for mineral transportation include the use of GPS-enabled vehicles with RFID tagging for monitoring, pre-registration of mineral-carrying vehicles with government portals, and establishment of mine monitoring systems with geo-fencing."
#     },
#     "blockchain_use_in_mining": {
#         "patterns": ["blockchain use in mining", "how can blockchain help in mining", "benefits of blockchain in mining"],
#         "response": "Blockchain technology can enhance transparency, efficiency, and security in the mining industry by enabling real-time tracking of value chains and supply chains, facilitating self-declaration of grades, tracking materials from extraction to production, automating invoice reconciliation, improving traceability of reserves, and validating workflow/audit processes."
#     },
#     "blockchain_recommendations": {
#         "patterns": ["blockchain recommendations", "how to implement blockchain in mining", "recommendations for adopting blockchain in mining"],
#         "response": "The committee recommends the adoption of blockchain technology in the mining sector, starting with a pilot project for high-value minerals like gold, copper, zinc, etc. Learnings from the pilot project can inform the replication of the model in other mines and minerals."
#     },
#     "grade_drop_in_odisha": {
#         "patterns": ["grade drop in Odisha", "reasons for grade drop in Odisha", "observations on grade drop in Odisha"],
#         "response": "The committee observed a sharp drop in the grade of ores produced in Odisha after March 31, 2021. This warrants detailed study of field operations, and concerned state governments should refer such cases to the central government for further investigation by authorities like the Geological Survey of India."
#     },
#     "applicability_on_other_ores": {
#         "patterns": ["applicability on other ores", "how do recommendations apply to other ores", "impact on other ores"],
#         "response": "The recommendations provided, mainly for iron ore, can be applied to all other ores where royalty, auction premium, and other payments to the government are dependent on the grade of the ore."
#     },
#     "views_of_state_governments": {
#         "patterns": ["views of state governments on the final draft", "state governments' opinions on the final report", "feedback from state governments on the final draft"],
#         "response": "State governments have provided feedback on the final draft of the report:\n- Chhattisgarh: Appreciates the recommendations and agrees with them, subject to consultation with stakeholders for feasibility. Already implementing certain measures like Khanij Online system.\n- Karnataka: Agreed with the final draft.\n- Odisha: Provided observations on the final draft.\n- Jharkhand: Agreed with the final draft."
#     },
#     "famous_scientists": {
#         "patterns": ["Who are some famous scientists?", "Can you name some renowned scientists?", "Tell me about famous scientists.", "Give me examples of notable scientists.", "Which scientists are well-known?"],
#         "responses": ["Some famous scientists include Albert Einstein, Isaac Newton, Marie Curie, Charles Darwin, and Nikola Tesla.", "Renowned scientists include Stephen Hawking, Galileo Galilei, Ada Lovelace, and Richard Feynman.", "Famous scientists throughout history include Leonardo da Vinci, Thomas Edison, Jane Goodall, and Neil deGrasse Tyson."]
#     },
#     "capital_cities": {
#         "patterns": ["What are the capital cities of different countries?", "Tell me about capital cities around the world.", "Can you list the capitals of various countries?", "What are the capitals of some nations?", "Give me information about capital cities."],
#         "responses": ["Some capital cities include London (United Kingdom), Paris (France), Tokyo (Japan), Beijing (China), and Moscow (Russia).", "Capital cities around the world include Washington, D.C. (United States), Berlin (Germany), Rome (Italy), Brasília (Brazil), and Canberra (Australia).", "The capitals of various countries include New Delhi (India), Ottawa (Canada), Cairo (Egypt), Buenos Aires (Argentina), and Seoul (South Korea)."]
#     },
#   "sampling_process": {
#     "patterns": ["What is the sampling process in Odisha mines?", "How does the sampling process work in Odisha mines?", "Can you explain the sampling process in Odisha mines?", "Tell me about the sampling process in non-exempted mines in Odisha."],
#     "responses": ["In Odisha mines, the sampling process involves various steps such as stack creation, sample collection, and chemical analysis. It utilizes technology like mobile apps and augmented reality for accuracy and efficiency."]
#   },
#   "issues_observed": {
#     "patterns": ["What are the issues observed in the stacking and sampling process?", "What challenges are faced in the stacking and sampling process?", "What problems have been identified in the stacking and sampling process?"],
#     "responses": ["Several issues have been observed in the stacking and sampling process in Odisha mines, including challenges related to stack size, space requirements, sampling accuracy, and human intervention. The process is also time-consuming and lacks continuous monitoring."]
#   },
#   "technological_advancement": {
#     "patterns": ["How can the stacking and sampling process be improved?", "What improvements can be made to the stacking and sampling process?", "Are there any suggestions for enhancing the stacking and sampling process?"],
#     "responses": ["To improve the stacking and sampling process, there is a need for technological advancement and automation. This would streamline processes, enhance efficiency, and ensure compliance with regulatory requirements."]
#   },
#   "additional_note": {
#     "pattern": "Additional note to the report dated 11.11.2022 of the committee on misclassification of grades of different grades of iron ore and other minerals –reg.",
#     "responses": [
#       "Ministry of Mines constituted a committee to examine the issue of misclassification of grades of iron ore and other minerals, adversely affecting the revenue of State Governments and suggest measures for preventing misclassification.",
#       "Comments/suggestions of stakeholders were sought on the recommendations of the committee. Comments received were forwarded to the committee for further consideration.",
#       "A meeting of the committee was convened on 13.07.2023 to discuss the comments received.",
#       "The committee observed that State Governments of Jharkhand, Chhattisgarh, and Karnataka generally accepted/agreed to the recommendations, while the State Government of Odisha had existing IT-based systems.",
#       "Recommendations in the report are recommendatory, and it is up to the State Governments to implement them.",
#       "The committee suggests implementing its recommendations on a pilot basis and utilizing learnings to prevent misreporting/misclassification of grades of minerals.",
#       "Additional recommendations suggested by the committee:",
#       "1. Adequate redressal mechanism for variation between physical inspection and system-enabled results.",
#       "2. Provision to allow dispatch in case of non-functionality of any equipment in the entire system.",
#       "3. The recommendation of the committee can be of recommendatory nature, and the adoption can be left to the State Government.",
#       "4. The system may initially be implemented on a pilot basis through a PSU."
#     ]
#   },
#     "misclassification_guidelines": {
#         "patterns": ["misclassification guidelines", "guidelines for preventing misclassification", "iron ore misclassification prevention", "mineral misclassification prevention"],
#         "responses": ["The guidelines aim to prevent misclassification of different grades of iron ore and other minerals, ensuring accurate assessment of Average Sale Price (ASP) and proper collection of statutory levies such as royalty."]
#     },
#     "mineral_sampling_analysis": {
#         "patterns": ["mineral sampling and analysis", "sampling and analysis guidelines", "mineral grade determination", "mineral analysis process"],
#         "responses": ["The guidelines emphasize the adoption of IT-enabled systems with minimal human intervention for sampling and analysis, integration of leaseholder systems with government monitoring systems, and encouragement of new technologies adoption through incentives."]
#     },
#     "continuous_online_analysis": {
#         "patterns": ["continuous online analysis", "online analyzer installation", "real-time analysis monitoring", "augur-based auto-samplers"],
#         "responses": ["Mandatory installation of continuous online analyzers for large mines, both mechanized and non-mechanized, use of augur-based auto-samplers for non-mechanized loading systems, and real-time monitoring of analysis results and CCTV surveillance are recommended."]
#     },
#     "royalty_statutory_payments": {
#         "patterns": ["royalty and statutory payments", "payment of royalty", "mineral dispatch payment", "seam analysis for royalty"],
#         "responses": ["Charging of royalty based on continuous analyzer analysis, monthly seam analysis for tolerance limit determination, and collection of other statutory payments are recommended."]
#     },
#     "transportation_monitoring": {
#         "patterns": ["transportation monitoring", "GPS-enabled vehicles", "RFID tagging", "vehicle tracking", "geo-fencing of mines"],
#         "responses": ["Use of GPS-enabled vehicles with RFID tagging for tracking and monitoring, pre-registration of mineral-carrying vehicles, and geo-fencing of mine boundaries are recommended."]
#     },
#     "blockchain_technology": {
#         "patterns": ["blockchain technology", "blockchain for mineral tracking", "blockchain for transparency", "blockchain for accountability"],
#         "responses": ["Implementation of blockchain for transparent tracking of mineral transactions, self-declaration of grades, invoice reconciliation, and potential pilot projects for high-value minerals are recommended."]
#     },
#     "applicability_to_other_ores": {
#         "patterns": ["applicability to other ores", "guidelines for other minerals", "royalty guidelines for other minerals", "grade-based payments for other ores"],
#         "responses": ["The guidelines are applicable to all ores where payments to the government depend on ore grade."]
#     },
#     "additional_recommendations": {
#         "patterns": ["additional recommendations", "misclassification prevention suggestions", "redressal mechanism", "dispatch provision for equipment failure"],
#         "responses": ["Establishment of a redressal mechanism for variation in inspection results, provision for dispatch in case of equipment non-functionality, and pilot implementation through a PSU with cross-verification of accuracy and detection limits are recommended."]
#     },






#      "mineral_rules": {
#         "patterns": [
#             "What are the Mineral Conservation and Development Rules?",
#             "Tell me about the Mineral Conservation and Development Rules, 2017.",
#             "Explain the amendments made in the Mineral Conservation and Development Rules, 2024.",
#             "When do the Mineral Conservation and Development (Amendment) Rules, 2024 come into force?",
#             "What is mentioned in rule 4 of the Mineral Conservation and Development Rules, 2017?",
#             "Can you provide information on reconnaissance permits and prospecting licenses?",
#             "How long does a holder of a reconnaissance permit or prospecting license have to submit a scheme?",
#             "Tell me about the modifications required for exploration licenses.",
#             "What is sub-section (11) of section 10BA in the Mineral Conservation and Development Rules?"
#         ],
#         "responses": [
#             "The Mineral Conservation and Development Rules govern the mining sector. The latest amendments are in the Mineral Conservation and Development (Amendment) Rules, 2024.",
#             "The Mineral Conservation and Development Rules, 2017, were amended by the Mineral Conservation and Development (Amendment) Rules, 2024.",
#             "The Mineral Conservation and Development (Amendment) Rules, 2024, came into force on the date of their publication in the Official Gazette.",
#             "In rule 4 of the Mineral Conservation and Development Rules, 2017, changes include the submission of a scheme for reconnaissance or prospecting within ninety days.",
#             "Reconnaissance permit and prospecting license holders must submit a scheme for operations within ninety days of obtaining the permit or license.",
#             "For exploration licenses, a modified scheme must be submitted after three years, indicating how the licensee plans to continue operations in the retained area.",
#             "Sub-section (11) of section 10BA relates to the retention of areas under exploration licenses.",
#         ]
#     },

# "mineral_rules_amendments": {
#         "patterns": [
#             "What are the Mineral (Auction) Amendment Rules, 2024?",
#             "Tell me about the recent amendments in the Mineral (Auction) Rules.",
#             "When do the Mineral (Auction) Amendment Rules, 2024, come into force?",
#             "Explain the changes in rule 2 of the Mineral (Auction) Rules, 2015.",
#             "What is the definition of 'auction premium' in the amended rules?",
#             "How is exploration license granted for minerals in the Seventh Schedule?",
#             "Can a bidder submit more than one bid in an auction?",
#             "Define 'affiliate' in the context of bidding.",
#             "What is the impact of the amendments on upfront payment for preferred bidders?",
#             "Tell me about the modifications in rule 19 regarding performance security.",
#             "Is there a limit on the performance security for holders of composite licenses?",
#             "What is mentioned in the new Rule 19A?"
#         ],
#         "responses": [
#             "The Mineral (Auction) Amendment Rules, 2024, are the recent amendments to the Mineral (Auction) Rules, 2015.",
#             "The Mineral (Auction) Amendment Rules, 2024, come into force upon publication in the Official Gazette.",
#             "Rule 2 of the Mineral (Auction) Rules, 2015, has been amended to include a definition of 'auction premium' and modify references in various clauses.",
#             "In the amended rules, 'auction premium' is defined as the amount payable by the lessee under sub-rule (2) of rule 13.",
#             "Exploration licenses for minerals in the Seventh Schedule are granted as specified in Chapter III A.",
#             "No, a bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
#             "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
#             "For preferred bidders selected after the commencement of the Mineral (Auction) Amendment Rules, 2024, the upfront payment should not exceed five hundred crore rupees.",
#             "Rule 19 has provisos specifying the maximum performance security amounts for preferred bidders and holders of composite licenses.",
#             "For preferred bidders selected after the commencement of the Mineral (Auction) Amendment Rules, 2024, the performance security limit should not exceed two hundred and fifty crore rupees.",
#             "A new rule, Rule 19A, has been introduced after Rule 19."
#         ]
#     },
#  "exploration_license": {
#         "patterns": [
#             "What is the process for obtaining an exploration license?",
#             "Tell me about the auction process for exploration licenses.",
#             "What are the prerequisites for initiating the auction of an exploration license?",
#             "Explain the role of the committee in identifying blocks for auction.",
#             "How does the State Government decide on areas for auction?",
#             "Can individuals submit proposals for exploration licenses?",
#             "What is the eligibility criteria for participating in the auction?",
#             "How is eligibility determined for exploration license bids?",
#             "Can a bidder submit more than one bid in an auction?",
#             "Define 'affiliate' in the context of exploration license bidding.",
#             "What are the exclusion criteria for identifying blocks for auction?",
#             "How does the Central Government approve auction recommendations?",
#             "Tell me about the requirements for participating in exploration license auctions.",
#             "What is the significance of Schedule I in the exploration license process?",
#             "Define terms like 'associate company,' 'control,' and 'subsidiary company' as per the Companies Act, 2013."
#         ],
#         "responses": [
#             "The process for obtaining an exploration license involves an auction initiated by the State Government.",
#             "Prerequisites for initiating the auction of an exploration license include submitting proposals and available geoscience data.",
#             "A committee, including members from various departments, identifies blocks for auction based on geological information.",
#             "The State Government decides on areas for auction through the committee, considering geological reports and data repositories.",
#             "Yes, individuals can submit proposals for exploration licenses by following the format specified in Schedule V.",
#             "Eligibility for participating in the auction is determined based on section 5 requirements and terms in Schedule I.",
#             "A bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
#             "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
#             "Exclusion criteria for identifying blocks include areas with existing concessions, ongoing tender processes, or certain exploration operations.",
#             "The Central Government approves auction recommendations within a specified time frame.",
#             "Requirements for participating in exploration license auctions are outlined in section 5 and Schedule I.",
#             "Schedule I specifies terms and conditions for eligibility and participation in exploration license auctions.",
#             "'Associate company,' 'control,' and 'subsidiary company' have the same meanings as assigned in the Companies Act, 2013."
#         ]
#     },
#  "exploration_license": {
#         "patterns": [
#             "What is the process for obtaining an exploration license?",
#             "How is the auction process for exploration licenses conducted?",
#             "Tell me about the prerequisites for initiating an exploration license auction.",
#             "Explain the role of the committee in identifying blocks for auction.",
#             "How does the State Government decide on areas for auction?",
#             "Can individuals submit proposals for exploration licenses?",
#             "What are the eligibility criteria for participating in the auction?",
#             "How is eligibility determined for exploration license bids?",
#             "Can a bidder submit more than one bid in an exploration license auction?",
#             "Define 'affiliate' in the context of exploration license bidding.",
#             "What are the exclusion criteria for identifying blocks for auction?",
#             "How does the Central Government approve auction recommendations?",
#             "Tell me about the requirements for participating in exploration license auctions.",
#             "What is the significance of Schedule I in the exploration license process?",
#             "Define terms like 'associate company,' 'control,' and 'subsidiary company' as per the Companies Act, 2013."
#         ],
#         "responses": [
#             "The process for obtaining an exploration license involves an auction initiated by the State Government.",
#             "Prerequisites for initiating the auction of an exploration license include submitting proposals and available geoscience data.",
#             "A committee, including members from various departments, identifies blocks for auction based on geological information.",
#             "The State Government decides on areas for auction through the committee, considering geological reports and data repositories.",
#             "Yes, individuals can submit proposals for exploration licenses by following the format specified in Schedule V.",
#             "Eligibility for participating in the auction is determined based on section 5 requirements and terms in Schedule I.",
#             "A bidder can submit only one bid in an auction, and affiliates are restricted from submitting bids in the same auction.",
#             "'Affiliate' refers to a person who controls, is controlled by, is under common control with, is an associate company of, or is a subsidiary company of the bidder.",
#             "Exclusion criteria for identifying blocks include areas with existing concessions, ongoing tender processes, or certain exploration operations.",
#             "The Central Government approves auction recommendations within a specified time frame.",
#             "Requirements for participating in exploration license auctions are outlined in section 5 and Schedule I.",
#             "Schedule I specifies terms and conditions for eligibility and participation in exploration license auctions.",
#             "'Associate company,' 'control,' and 'subsidiary company' have the same meanings as assigned in the Companies Act, 2013."
#         ]
#     },
#     "electronic_auction": {
#         "patterns": [
#             "How is an auction conducted for exploration licenses electronically?",
#             "Tell me about the platform used for electronic exploration license auctions.",
#             "What are the bidding parameters for exploration licenses?",
#             "Explain the electronic auction process for exploration licenses.",
#             "How does the State Government specify the ceiling price for auction premiums?",
#             "What is the bidding process for exploration licenses?",
#             "Tell me about the notice inviting tender for exploration license auctions.",
#             "What are the requirements for submitting a technical bid in exploration license auctions?",
#             "Define bid security and its amount for exploration license auctions.",
#             "How are technically qualified bidders determined in exploration license auctions?",
#             "What is the process for the second round of online electronic auction?",
#             "How does the Central Government conduct auctions for exploration licenses?",
#             "What information does the State Government provide to the Central Government regarding exploration license auctions?",
#             "How is the preferred bidder determined after a successful auction?"
#         ],
#         "responses": [
#             "Exploration license auctions are conducted exclusively through online electronic auction platforms.",
#             "State Governments can use online platforms meeting specified technical and security requirements.",
#             "The State Government specifies a maximum percentage share ('ceiling price') of the auction premium for future lessees.",
#             "The auction is a descending reverse online electronic auction with two rounds.",
#             "Bidders quote a percentage share of the auction premium, and the one quoting the minimum percentage becomes the preferred bidder.",
#             "State Government issues a notice inviting tender, providing details on the area under auction and available geoscience data.",
#             "The tender document includes information on the identified area, geoscience data, and the bidding process.",
#             "Bidders submit a technical bid and an initial price offer in the first round of the auction.",
#             "Bid security is required based on the area size.",
#             "Only technically qualified bidders proceed to the second round, submitting a final price offer.",
#             "The lowest bidder in the second round becomes the preferred bidder.",
#             "The State Government informs the Central Government about various stages of exploration license auctions.",
#             "Central Government follows the same auction rules as applicable to State Governments for exploration licenses.",
#             "Upon successful completion of the auction, the Central Government informs the State Government about the preferred bidder."
#         ]
#     },
#  "exploration_license": {
#         "patterns": ["What is an exploration license?", "How can I obtain an exploration license?", "Tell me about exploration licenses."],
#         "responses": ["An exploration license allows individuals to initiate the process of obtaining mining rights for specified minerals. The process involves submitting a proposal to the State Government and participating in an auction."]
#     },
#     "grant_process": {
#         "patterns": ["How is the exploration license granted?", "Explain the grant process for exploration licenses."],
#         "responses": ["The exploration license grant process involves submitting a proposal, participating in an auction, and fulfilling conditions such as obtaining approvals and submitting a reconnaissance or prospecting scheme."]
#     },
#     "performance_security": {
#         "patterns": ["What is performance security?", "Tell me about the performance security.", "Why is performance security required?"],
#         "responses": ["Performance security is a financial guarantee provided by the preferred bidder. It ensures commitment to the exploration license process. It may be forfeited in case of non-compliance with specified conditions."]
#     },
#     "payment_process": {
#         "patterns": ["How is payment handled for exploration licenses?", "Tell me about the payment process for exploration licenses."],
#         "responses": ["Payment to exploration licensees involves receiving a percentage share from the auction premium deposited by the future lessee. The share is payable for the entire mining lease period or until resource exhaustion."]
#     },
#     "share_transfer": {
#         "patterns": ["Can the exploration license share be transferred?", "Tell me about transferring the exploration license share."],
#         "responses": ["Yes, the exploration license share can be transferred to another entity with State Government approval."]
#     },
#     "timeline_conditions": {
#         "patterns": ["What are the timelines and conditions for exploration licenses?", "Explain the timelines and conditions during exploration licenses."],
#         "responses": ["Timelines include submission of performance security, fulfillment of conditions, and surrender options. Conditions involve compliance, approvals, and submission of a prospecting scheme."]
#     },
#     "geological_exploration": {
#         "patterns": ["What is geological exploration?", "Explain the geological exploration process during exploration licenses."],
#         "responses": ["Geological exploration involves studying the mineral content of the licensed area. Exploration licensees submit periodic reports to the State Government and the Indian Bureau of Mines."]
#     },
#     "termination_conditions": {
#         "patterns": ["Under what conditions can exploration licenses be terminated?", "Explain the conditions for exploration license termination."],
#         "responses": ["Exploration licenses can be terminated if the licensee fails to complete operations or establish mineral contents within the specified period. The State Government may take appropriate actions."]
#     },
#  "exploration_license_auction": {
#         "patterns": ["What are the rules for exploration license auction?", "Tell me about auction rules for exploration license.", "How is exploration license auction conducted?"],
#         "responses": ["The auction for exploration license is conducted through an online electronic platform. The process involves various steps such as issuing tender documents, submitting bids, and a descending reverse online auction."]
#     },
#     "mining_lease_auction": {
#         "patterns": ["How is mining lease auction conducted?", "Tell me about auction rules for mining lease.", "What are the terms for mining lease auction?"],
#         "responses": ["Mining lease auction follows the rules specified in Chapter II of the regulations. The State Government initiates the auction within six months of receiving the geological report from the exploration licensee. The preferred bidder is selected within one year from the date of receiving the geological report."]
#     },
#     "preferred_bidder_selection": {
#         "patterns": ["How is the preferred bidder selected?", "What factors determine the preferred bidder?", "Tell me about the selection of the preferred bidder."],
#         "responses": ["The preferred bidder is selected based on various factors, including the geological report submitted by the exploration licensee. If the preferred bidder is not selected within the specified period, the State Government compensates the exploration licensee for the incurred expenditure."]
#     },
#     "termination_or_surrender": {
#         "patterns": ["What happens in case of termination or surrender of a mining lease?", "Tell me about the consequences of lease termination.", "Explain the process if a lease is surrendered."],
#         "responses": ["In case of termination, lapse, or surrender of a mining lease, the State Government provides an opportunity to the exploration licensee to obtain the lease in the same area at the auction premium discovered earlier."]
#     },
#     "participation_in_auction": {
#         "patterns": ["Can the exploration licensee participate in the auction?", "Are there restrictions on participation in mining lease auction?", "Tell me about eligibility for auction participation."],
#         "responses": ["The exploration licensee is not prohibited from participating in the auction for the mining lease. However, they need to fulfill the eligibility conditions specified in rule 6."]
#     },
#     "tender_document_details": {
#         "patterns": ["What information is in the tender document?", "Tell me about the contents of the tender document.", "What details are provided in the auction tender document?"],
#         "responses": ["The tender document contains details such as raw data and bore-hole cores generated during prospecting operations. Additionally, it includes ownership structure or shareholding details of the exploration licensee."]
#     },
#     "related_party_declaration": {
#         "patterns": ["What is a related party declaration?", "Why is declaring related party important?", "Explain related party declaration in mining lease auction."],
#         "responses": ["A bidder participating in the auction must declare to the government if they are a related party of the exploration licensee. This declaration ensures transparency in the auction process."]
#     },
#  "exploration_license": {
#         "patterns": [
#             "Tell me about exploration licenses",
#             "What is the process for auctioning exploration licenses?",
#             "How can I participate in the auction for exploration licenses?",
#             "Explain the requirements for exploration license proposals",
#             "Details needed for proposing an exploration license"
#         ],
#         "responses": [
#             "Exploration licenses are granted for the prospecting of mineral-rich areas. The process involves submitting a proposal to the Mining and Geology Department.",
#             "Auctioning exploration licenses follows specific rules. The State Government initiates the process within six months of receiving a geological report.",
#             "To participate in the auction, you need to submit a proposal with details about the location, mineral potential, and relevant documentation.",
#             "Requirements for exploration license proposals include applicant information, area details, mineral potential, and necessary documentation.",
#             "For proposing an exploration license, you need to provide applicant details, location specifics, details of mineral potential, and required documentation."
#         ]
#     },
# "amendment_rules": {
#         "patterns": ["Tell me about the Mineral Conservation and Development (Amendment) Rules, 2024", "What are the changes in the rules?", "Explain the amendment rules"],
#         "responses": [
#             "The Mineral Conservation and Development (Amendment) Rules, 2024 were introduced by the Central Government to amend the Mineral Conservation and Development Rules, 2017.",
#             "These rules came into force upon their publication in the Official Gazette.",
#             "One significant change is in Rule 4, where the submission of schemes for reconnaissance or prospecting is now required within ninety days from the date of execution of the permit or license.",
#             "In Rule 5, changes include the insertion of 'or both' after 'reconnaissance or prospecting' and the addition of exploration license in various places.",
#             "Rule 9A imposes restrictions on the disclosure of information, schemes, and reports by the holder of an exploration license."
#         ]
#     },
#     "submission_of_schemes": {
#         "patterns": ["How should I submit a scheme for reconnaissance or prospecting?", "Tell me about the submission of schemes", "What is required for submitting a scheme?"],
#         "responses": [
#             "Every holder of a reconnaissance permit or prospecting license should submit a scheme within ninety days from the date of execution of the permit or license.",
#             "For exploration license, a modified scheme should be submitted after three years from the date of execution.",
#             "The scheme should indicate the manner in which the licensee proposes to carry out reconnaissance or prospecting operations in the covered area."
#         ]
#     },
#     "half_yearly_reports": {
#         "patterns": ["Explain the half-yearly reports", "What should be included in the half-yearly report?", "Tell me about reporting obligations"],
#         "responses": [
#             "Every holder of a reconnaissance permit, prospecting license, composite license, or exploration license should submit a half-yearly report to the Regional Controller or the authorized officer and the State Government.",
#             "The report should cover operations from January to June and July to December each year.",
#             "Exploration licensees must also submit a geological report within three months of completing operations, identifying areas suitable for a mining lease."
#         ]
#     },
#     "restriction_on_disclosure": {
#         "patterns": ["What restrictions are there on disclosing information?", "Tell me about disclosure restrictions", "Explain information disclosure rules"],
#         "responses": [
#             "The holder of an exploration license is restricted from disclosing information, schemes, and reports to anyone other than the specified government or authorities without prior approval from the Central Government."
#         ]
#     },
#     "mining_plan_review": {
#         "patterns": ["When is a mining plan review required?", "Tell me about the review of mining plans", "Explain the mining plan review process"],
#         "responses": [
#             "Mining or mineral processing operations may require a mining plan review if discontinued for a period exceeding specified days.",
#             "The holder should submit a notice to the authorized officer and the State Government under rule 28.",
#             "The exact requirements depend on the specific rules mentioned in the respective sections."
#         ]
#     },
#     "schedule_amendments": {
#         "patterns": ["What amendments are there in Schedule I?", "Tell me about changes in Schedule I", "Explain the amendments to forms"],
#         "responses": [
#             "Amendments in Schedule I include inserting 'or exploration license' wherever 'composite license' is mentioned.",
#             "Forms such as Form-A, Form-B, Form-H, Form-I, Form-J, Form-K, and Form-N now include 'or exploration license'.",
#             "Important instructions for filling the form have also been updated in Form-B."
#         ]
#     },
#  "violation_penalty": {
#         "patterns": ["What are the penalties for rule violations?", "Tell me about penalties under Rule 45", "Explain the fines for non-compliance"],
#         "responses": [
#             "Under Rule 45, the amount to be paid in case of violation depends on the nature of the violation.",
#             "For non-submission or incomplete/wrong/false information in monthly returns (Form F1, F2, F3), the penalty ranges from ₹5,000 to ₹10,000 per day, depending on the leased area and production capacity.",
#             "Similar penalties apply for violations in annual returns (Form G1, G2, G3), monthly returns (Form L), and annual returns (Form M).",
#             "Specific rules (e.g., Rule 11, Rule 12, Rule 18, etc.) also have associated fines for contravention.",
#             "Feel free to ask about a specific rule or type of violation for more details."
#         ]
#     },
#     "specific_rule_penalty": {
#         "patterns": ["What is the penalty for violating Rule 11?", "Tell me about fines for Rule 18 violations", "Explain penalties under Rule 28"],
#         "responses": [
#             "Certainly! Here are some examples of fines for specific rules:",
#             "- Rule 11 (Mining operations under mining lease): ₹1,000 per day, subject to a maximum of ₹5,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes.",
#             "- Rule 18 (Beneficiation studies to be carried out): ₹1,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes, ₹5,00,000 for other cases.",
#             "- Rule 28 (Notice of temporary discontinuance of work in mines and obligations of lease holders): ₹1,00,000 for leases up to 25 hectares and having per annum approved production capacity up to 2 lakh tonnes, ₹5,00,000 for other cases."
#         ]
#     },
#  "mineral_rules_amendment": {
#         "patterns": ["Tell me about the Minerals (Evidence of Mineral Contents) Amendment Rules, 2024", "Explain the recent amendments in mineral rules", "What changes have been made in the Minerals (Evidence of Mineral Contents) Rules, 2015?"],
#         "responses": [
#             "The Minerals (Evidence of Mineral Contents) Amendment Rules, 2024 have been introduced with some key changes.",
#             "One significant change is in Rule 2, where additional criteria have been specified for minerals with a grade equal to or more than the threshold value under the Atomic Minerals Concession Rules, 2016.",
#             "Rule 5 has been updated to include 'section 11D' in addition to 'section 11'.",
#             "Rule 7 now has provisions specific to minerals in Part D of the First Schedule to the Act. Proposals for such minerals are to be submitted to the Central Government, and a committee will assess mineral potentiality for blocks proposed by individuals.",
#             "Schedule III includes information on where proposals should be submitted - either to the State Government or the Central Government."
#         ]
#     },
#     "submission_procedures": {
#         "patterns": ["How do I submit a proposal under the Minerals (Evidence of Mineral Contents) Rules?", "Tell me about the submission procedures for mineral proposals"],
#         "responses": [
#             "To submit a proposal under the Minerals (Evidence of Mineral Contents) Rules:",
#             "- Follow the guidelines in Schedule III of the rules.",
#             "- Depending on your case, submit your proposal to either the State Government or the Central Government.",
#             "- Ensure that you comply with the relevant sections mentioned in the amendments, such as section 11 or section 11D.",
#             "- For minerals in Part D of the First Schedule, proposals need to be submitted to the Central Government and evaluated by a committee.",
#             "Feel free to ask if you have specific questions about the submission process."
#         ]
#     },
# "mineral_rules": {
#         "patterns": ["What are the Mineral Conservation and Development Rules?", "Tell me about the Minerals (Evidence of Mineral Contents) Rules, 2015.", "Explain the amendments in the Minerals (Evidence of Mineral Contents) Amendment Rules, 2024.", "When do the amended rules come into force?", "Who approves proposals for minerals specified in Part D of the First Schedule to the Act?"],
#         "responses": ["The Mineral Conservation and Development Rules govern mining activities.", "The Minerals (Evidence of Mineral Contents) Rules, 2015, specify requirements for mineral exploration.", "The amendments in 2024 introduce changes to how mineral potentiality is identified and submitted.", "The amended rules came into force on January 21, 2024.", "Proposals for minerals in Part D are approved by the Central Government."]
#     },
#     "amendment_details": {
#         "patterns": ["What changes were made in Rule 2?", "Explain the amendments in Rule 5.", "Tell me about the committee mentioned in Rule 7(1B).", "What destinations are mentioned in Rule 3 of Schedule III?", "When were the Principal Rules last amended?"],
#         "responses": ["Rule 2 was amended to include specifications for minerals based on grade.", "Rule 5 was amended to include references to section 11D.", "Rule 7(1B) introduces a committee to identify mineral potentiality.", "Rule 3 in Schedule III clarifies submission destinations: State Government or Central Government.", "The Principal Rules were last amended on December 14, 2021."]
#     },
#     "conclusion": {
#         "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
#         "responses": ["The amendments focus on specifying mineral criteria, submitting proposals to the Central Government, and establishing a committee for mineral potentiality.", "The conclusion highlights changes related to minerals, submission destinations, and the effective date of the amendments."]
#     },
#  "atomic_minerals_rules": {
#         "patterns": ["What are the Atomic Minerals Concession Rules, 2016?", "Tell me about the amendments in the Atomic Minerals Concession (Amendment) Rules, 2023.", "When did the amended rules come into force?", "What is the penalty for contravening the specified rules?"],
#         "responses": ["The Atomic Minerals Concession Rules, 2016, regulate the mining of atomic minerals.", "The Amendment Rules of 2023 introduce changes in the penalty provisions and deposit exploration details.", "The amended rules came into force on September 22, 2023.", "The penalty for contravening specified rules may include imprisonment up to two years, a fine up to five lakhs, or both."]
#     },
#     "amendment_details": {
#         "patterns": ["What changes were made in Rule 37?", "Explain the substituted entries in Schedule B, Part III.", "Tell me about the exploration details for Rare metal and REE deposits in pegmatites, reefs, and veins/pipes (Serial Number III)."],
#         "responses": ["Rule 37 introduces new provisions for penalties related to specified rules.", "In Schedule B, Part III, Serial Number III for Rare metal and REE deposits in pegmatites, scout drilling/pitting/trenching is required at 10 to 25 pits/trenches per sq.km.", "Exploration details for Rare metal and REE deposits in pegmatites include scout drilling/pitting/trenching and exploratory open pit or boreholes at specified intervals."]
#     },
#     "conclusion": {
#         "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
#         "responses": ["The amendments focus on introducing penalties for contraventions, updating deposit exploration details, and specifying the rules related to penalties.", "The conclusion highlights changes in penalty provisions, exploration requirements, and the effective date of the amendments."]
#     },
#  "mineral_auction_rules": {
#         "patterns": ["What are the Mineral (Auction) Rules, 2015?", "Explain the amendments in the Mineral (Auction) Amendment Rules, 2023.", "When did the amended rules come into force?", "What changes were made in rule 5, sub-rule (2)?", "Tell me about the new rules 9B and 17B."],
#         "responses": [
#             "The Mineral (Auction) Rules, 2015, govern the auctioning of minerals.",
#             "The Amendment Rules of 2023 introduce changes in rule 5, sub-rule (2) and add rules 9B and 17B.",
#             "The amended rules came into force on September 1, 2023.",
#             "Rule 5, sub-rule (2) now includes a proviso allowing the use of land details from the Prime Minister Gati Shakti - National Master Plan or state land record portals for land classification.",
#             "Rule 9B relates to the conduct of auction of mining leases by the Central Government, and Rule 17B relates to the conduct of auction of composite licenses by the Central Government."
#         ]
#     },
#     "rule_details": {
#         "patterns": ["Explain rule 9B.", "What are the key points of rule 17B?", "Tell me about the provisions of conducting an auction by the Central Government under section 11D."],
#         "responses": [
#             "Rule 9B outlines the procedure for conducting auctions of mining leases by the Central Government, including the intimation of details by the State Government, receipt of geological reports, termination of leases, and the role of the Central Government.",
#             "Rule 17B details the procedure for the Central Government to conduct auctions of composite licenses, involving the intimation of details by the State Government, receipt of geological reports, termination of licenses, and the role of the Central Government.",
#             "For conducting an auction under section 11D, the Central Government follows rules 5 to 9 (for mining leases) or rules 16 and 17 (for composite licenses) and informs the State Government of the preferred bidder upon successful completion."
#         ]
#     },
#     "conclusion": {
#         "patterns": ["What is the conclusion of the notification?", "Summarize the key points of the amendments.", "What are the main focuses of the amendments?"],
#         "responses": [
#             "The amendments focus on incorporating land details from the Gati Shakti platform, updating rules related to land classification, and introducing procedures for the Central Government to conduct auctions of mining leases and composite licenses."
#         ]
#     },
# "minerals_concession_rules": {
#         "patterns": ["What are the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2016?", "When did these rules come into force?", "What minerals do these rules apply to?", "Define 'railway' and 'run-of-mine'."],
#         "responses": [
#             "The Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2016, were established under the Mines and Minerals (Development and Regulation) Act, 1957.",
#             "These rules came into force on the date of their publication in the Official Gazette.",
#             "These rules apply to all minerals, except minor minerals defined under Section 3(e) and minerals listed in Part A and Part B of the First Schedule to the Act.",
#             "In these rules, 'railway' and 'run-of-mine' are defined as per the Indian Railways Act and refer to raw, unprocessed material obtained from the mineralized zone of a lease area, respectively."
#         ]
#     },
#     "definitions": {
#         "patterns": ["Define 'illegal mining,' 'mineral concession,' 'run-of-mine,' etc.", "Explain the meaning of 'value of estimated resources.'"],
#         "responses": [
#             "'Illegal mining' refers to unauthorized reconnaissance, prospecting, or mining operations in an area without the required mineral concession.",
#             "'Mineral concession' includes reconnaissance permit, non-exclusive reconnaissance permit, prospecting licence, prospecting licence-cum-mining lease, or a mining lease.",
#             "'Run-of-mine' is the raw material obtained after blasting or digging from the mineralized zone of a lease area.",
#             "'Value of estimated resources' is the product of the estimated quantity of mineral resources granted by a concession and the average price per metric tonne of the mineral, as published by the Indian Bureau of Mines for the relevant state, over the preceding twelve months."
#         ]
#     },
#     "rights_of_existing_holders": {
#         "patterns": ["What are the rights of the existing holders of mineral concessions?", "Explain the procedure for a holder of a reconnaissance permit to apply for a prospecting licence."],
#         "responses": [
#             "Existing holders of mineral concessions have specific rights under these rules. For example, a holder of a reconnaissance permit may apply for a prospecting licence under certain conditions.",
#             "For a holder of a reconnaissance permit, the rules outline a procedure for applying for a prospecting licence, including the application format, acknowledgement process, fees, and the role of the State and Central Governments."
#         ]
#     },
#     "renewal_of_prospecting_licence": {
#         "patterns": ["How can one renew a prospecting licence?", "What information needs to be provided for the renewal of a prospecting licence?"],
#         "responses": [
#             "The renewal of a prospecting licence involves submitting an application ninety days before expiry, providing a statement with reasons, a report on prospecting operations, expenditure details, and justification for additional time.",
#             "The State Government acknowledges the renewal application, and the process includes a non-refundable fee, possible condonation of delay, and timely disposal by the State Government."
#         ]
#     },
#  "prospecting_to_mining": {
#         "patterns": [
#             "How can I apply for a mining lease?",
#             "What are the steps to obtain a mining lease from a prospecting licence?",
#             "Tell me about mining lease application process.",
#             "What is required for mining lease after a prospecting licence?"
#         ],
#         "responses": [
#             "To obtain a mining lease from a prospecting licence, you need to follow these steps:",
#             "1. Submit an application for a mining lease within three months after the prospecting licence expiry.",
#             "2. The State Government will acknowledge your application and may require a non-refundable fee of Rs. 5 lakhs per sq. km.",
#             "3. Fulfill conditions specified in sub-clause (i) to sub-clause (iv) of clause (b) of sub-section (2) of Section 10A.",
#             "4. The State Government decides within 60 days, may forward to Central Government for approval.",
#             "5. Once approved, obtain necessary consents, approvals, permits, provide performance security, and sign an agreement with the State Government.",
#             "6. The State Government executes a mining lease deed within 90 days. If not executed, the order may be revoked."
#         ]
#     },
#     "section_100_c": {
#         "patterns": [
#             "What are the rights under Section 100(c)?",
#             "Tell me about the rights for mining lease grant under Section 100(c).",
#             "Explain Section 100(c) of the Mines and Minerals Act.",
#             "What happens if conditions in the letter of intent or previous approval are not fulfilled?"
#         ],
#         "responses": [
#             "Under Section 100(c) of the Mines and Minerals Act:",
#             "1. The applicant submits a letter of compliance for the conditions mentioned in the letter of intent or previous approval.",
#             "2. The State Government issues an order for grant of the mining lease within 60 days, subject to condition verification.",
#             "3. If conditions are not fulfilled, the State Government may refuse to grant a mining lease.",
#             "4. Upon issuance of an order, the applicant must furnish a performance security to the State Government and sign an agreement.",
#             "5. The mining lease should be executed and registered on or before January 11, 2017, else the right is forfeited."
#         ]
#     },
#  "prospecting_lease_through_auction": {
#         "patterns": [
#             "Tell me about the composite license granted through auction.",
#             "What is the format for prospecting license deed under the Mineral (Auction) Rules, 2015?",
#             "Explain the mining lease deed for successful bidders under Mineral (Auction) Rules."
#         ],
#         "responses": [
#             "1. The prospecting license deed for the composite license is in the format specified in Schedule V.",
#             "2. Mining lease deed for successful bidders under Mineral (Auction) Rules, 2015, is in the format specified in Schedule VII."
#         ]
#     },
#     "renewal_of_prospecting_license": {
#         "patterns": [
#             "How can I renew a prospecting license?",
#             "Tell me about the renewal process for a composite license prospecting stage.",
#             "What is the fee for renewing a prospecting license?"
#         ],
#         "responses": [
#             "To renew a prospecting license:",
#             "1. Apply at least ninety days before the expiry with reasons, details of operations, expenditure, man-days, and justification for additional time.",
#             "2. State Government acknowledges the renewal application within three days.",
#             "3. Pay a non-refundable fee of Rs. 1000 per sq. km.",
#             "4. State Government may condone delay if applied before prospecting license stage expiry.",
#             "5. State Government decides on renewal before the prospecting license expires."
#         ]
#     },
#     "terms_and_conditions_of_licenses": {
#         "patterns": [
#             "What are the terms and conditions of a prospecting license?",
#             "Explain the conditions for licensees under the Mines and Minerals Act.",
#             "Tell me about the licensee's responsibilities and obligations."
#         ],
#         "responses": [
#             "1. Licensee may win minerals within specified limits without payment or on payment of royalty.",
#             "2. Licensee can carry away minerals for chemical, metallurgical, ore-dressing, and test purposes with written permission.",
#             "3. Licensee convicted of illegal mining may have the license canceled and performance security forfeited.",
#             "4. Licensee must report the discovery of any mineral within sixty days.",
#             "5. Licensee needs to comply with Act and rules, restore affected land, maintain accurate accounts, and follow specific conditions."
#         ]
#     },
#  "mining_lease": {
#         "patterns": [
#             "What are the conditions for a mining lease?",
#             "Tell me about mining lease terms",
#             "Explain the obligations in a mining lease",
#             "What are the restrictions on mining operations?",
#             "How should a lessee report accidents in a mining lease?"
#         ],
#         "responses": [
#             "In a mining lease, the lessee must pay yearly dead rent, commence operations within two years, and follow restrictions on mining activities.",
#             "Mining lease terms involve payment obligations, environmental responsibilities, and government rights.",
#             "Obligations include payment of dead rent, surface rent, and water rate, along with maintaining accurate records and restoring affected landforms.",
#             "Restrictions include no mining within 50 meters from railways without permission and not interfering with public grounds or village roads.",
#             "Accidents in mining operations must be promptly reported to the Deputy Commissioner or Collector."
#         ]
#     },
#  "mining_lease": {
#         "patterns": [
#             "What are the conditions for a mining lease?",
#             "Tell me about mining lease terms",
#             "Explain the obligations in a mining lease",
#             "What are the restrictions on mining operations?",
#             "How should a lessee report accidents in a mining lease?"
#         ],
#         "responses": [
#             "In a mining lease, the lessee must pay yearly dead rent, commence operations within two years, and follow restrictions on mining activities.",
#             "Mining lease terms involve payment obligations, environmental responsibilities, and government rights.",
#             "Obligations include payment of dead rent, surface rent, and water rate, along with maintaining accurate records and restoring affected landforms.",
#             "Restrictions include no mining within 50 meters from railways without permission and not interfering with public grounds or village roads.",
#             "Accidents in mining operations must be promptly reported to the Deputy Commissioner or Collector."
#         ]
#     },
#     "prospecting_license": {
#         "patterns": [
#             "What is the process for renewal of a prospecting license?",
#             "Tell me about prospecting license conditions",
#             "What are the rights and responsibilities of a prospecting license holder?",
#             "Explain the restrictions on prospecting operations.",
#             "How does force majeure affect prospecting license terms?"
#         ],
#         "responses": [
#             "Prospecting license renewal requires an application with reasons, reports of prospecting operations, and justifications for additional time.",
#             "Conditions include the right to win minerals for testing, restoration of land, and compliance with Act and rules.",
#             "Rights involve winning and carrying minerals, restoration of landforms, and maintaining accurate accounts.",
#             "Restrictions include obtaining permission for clearing land, complying with rules, and reporting mineral discoveries.",
#             "Force majeure events may extend the prospecting license period, and delay due to force majeure is not considered a breach."
#         ]
#     },
#   "mining_lease_info": {
#         "patterns": ["What are the conditions of a mining lease?", "Tell me about mining lease rules", "Explain mining lease provisions", "What rights do mining lease holders have?", "How does pre-emption work in mining leases?", "What are the conditions for lease termination?", "Tell me about discovered minerals in a mining lease"],
#         "responses": [
#             "Mining lease conditions include payment of rents and royalties, surface rent, and water rate.",
#             "Mining lease holders have rights for mining operations, including working the mines, constructing infrastructure, obtaining materials, and more.",
#             "Pre-emption in mining leases allows the government to purchase discovered minerals from the lease holder.",
#             "Lease termination can occur due to default, illegal mining, or failure to comply with conditions.",
#             "Mining lease holders must follow specific conditions outlined by the State Government.",
#             "Discovered minerals in a mining lease can be disposed of after inclusion in the lease deed."
#         ]
#     },
#  "mining_lease_conditions": {
#         "patterns": ["What are the conditions of a mining lease?", "Tell me about mining lease rules", "Explain mining lease provisions", "What are the obligations of a mining lease holder?"],
#         "responses": [
#             "Mining lease conditions include payment of rents, royalties, surface rent, and water rate.",
#             "Lessee must commence mining operations within two years, conduct operations properly, and restore the landform after mining operations.",
#             "Mining lease holders should give preference to tribals and those displaced by mining operations.",
#             "Conditions also cover areas like pre-emption rights, storage of unutilized ores, employment preferences, and more."
#         ]
#     },
#     "mining_lease_discovered_minerals": {
#         "patterns": ["What happens if a new mineral is discovered?", "Tell me about discovered minerals in a mining lease", "Can a mining lease holder dispose of discovered minerals?"],
#         "responses": [
#             "Holder of a mining lease through auction may win and dispose of the mineral discovered after inclusion in the mining lease deed.",
#             "In case of a mining lease not granted through auction, the state government may exercise pre-emption rights and pay the holder the cost of production for the mineral.",
#             "Discovery of minerals not specified in the lease by the holder not granted through auction doesn't grant disposal rights; the state government may exercise pre-emption."
#         ]
#     },
#     "mining_lease_further_conditions": {
#         "patterns": ["What other conditions can be in a mining lease?", "Tell me about additional mining lease conditions", "What conditions can the state government impose for mineral development?"],
#         "responses": [
#             "Mining leases may contain conditions on payment modes, compensation for land damage, tree felling restrictions, surface operations, reporting accidents, indemnity to the government, and more.",
#             "State governments may impose further conditions for mineral development in the interests of the industry."
#         ]
#     },
#  "mining_plan": {
#         "patterns": ["What is a mining plan?", "How to get a mining plan approved?", "Tell me about the components of a mining plan", "Who approves the mining plan?", "Explain the process for mining plan modification", "What qualifications are required for preparing a mining plan?"],
#         "responses": [
#             "A mining plan is required for any mining operations and must be approved by the Indian Bureau of Mines or the State Government.",
#             "To get a mining plan approved, submit it to the designated officer along with the required fee. The Indian Bureau of Mines will then review the application.",
#             "A mining plan includes details like the lease hold area, geology, exploration program, mining methods, environmental management, and more.",
#             "The mining plan is approved by the Indian Bureau of Mines or the State Government as per the established system.",
#             "For modification, submit the proposed changes to the approving authority. The process follows the rules outlined in the mining plan regulations.",
#             "To prepare a mining plan, one needs a degree in mining engineering or a post-graduate degree in geology, along with five years of supervisory experience in mining."
#         ]
#     },
#  "mining_plan": {
#         "patterns": ["What is a mining plan?", "How to get a mining plan approved?", "Tell me about the components of a mining plan", "Who approves the mining plan?", "Explain the process for mining plan modification", "What qualifications are required for preparing a mining plan?"],
#         "responses": [
#             "A mining plan is required for any mining operations and must be approved by the Indian Bureau of Mines or the State Government.",
#             "To get a mining plan approved, submit it to the designated officer along with the required fee. The Indian Bureau of Mines will then review the application.",
#             "A mining plan includes details like the lease hold area, geology, exploration program, mining methods, environmental management, and more.",
#             "The mining plan is approved by the Indian Bureau of Mines or the State Government as per the established system.",
#             "For modification, submit the proposed changes to the approving authority. The process follows the rules outlined in the mining plan regulations.",
#             "To prepare a mining plan, one needs a degree in mining engineering or a post-graduate degree in geology, along with five years of supervisory experience in mining."
#         ]
#     },
#     "mining_operations": {
#         "patterns": ["What are the details of mining operations?", "Tell me about the methods of mining operation", "Explain environmental management in mining", "What is a progressive mine closure plan?"],
#         "responses": [
#             "Mining operations involve activities like excavation, drilling, blasting, waste handling, mineral beneficiation, and more.",
#             "Methods of mining operation include excavation techniques, drilling and blasting processes, mineral handling, and beneficiation.",
#             "Environmental management in mining includes baseline information, impact assessment, and mitigation measures.",
#             "A progressive mine closure plan outlines the steps for the closure of the mine in a phased manner as defined in the rules."
#         ]
#     },
#  "lease_conditions": {
#         "patterns": ["Tell me about the conditions in a mining lease", "What are the restrictions for felling trees in a leased area?", "Explain compensation for land damage in a mining lease", "How are accidents reported in mining operations?"],
#         "responses": [
#             "Conditions in a mining lease cover aspects like rent and royalties payment, compensation for land damage, restrictions on tree felling, and more.",
#             "Restrictions on felling trees in a leased area are outlined in the lease conditions imposed by the State Government.",
#             "Compensation for land damage is a part of the mining lease conditions and is specified based on the rules set by the State Government.",
#             "Accidents in mining operations are required to be reported as per the conditions outlined in the mining lease. The reporting process is crucial for safety measures."
#         ]
#     },
#  "mining_lease_expiry": {
#         "patterns": ["What happens on the expiry of a mining lease?", "Tell me about mining lease expiry", "Expiry of mining lease"],
#         "responses": ["After the expiry of a mining lease, it will be put up for auction following specified procedures."]
#     },
#     "right_of_first_refusal": {
#         "patterns": ["Explain the right of first refusal in mining leases", "What is the right of first refusal?", "How does right of first refusal work?"],
#         "responses": ["The holder of a mining lease for captive purposes has the right of first refusal at the time of auction after lease expiry."]
#     },
#     "lapsing_of_mining_lease": {
#         "patterns": ["What is the process of lapsing of a mining lease?", "When does a mining lease lapse?", "Lapsing of mining lease"],
#         "responses": ["A mining lease may lapse if mining operations don't commence within two years or are discontinued for two continuous years."]
#     },
#     "surrender_of_mining_lease": {
#         "patterns": ["How can a mining lease be surrendered?", "Tell me about surrendering a mining lease", "Surrender process of mining lease"],
#         "responses": ["The lessee may apply for the surrender of the entire area of the mining lease after giving a notice of not less than twelve calendar months from the intended date of surrender."]
#     },
#     "termination_of_mining_lease": {
#         "patterns": ["Under what circumstances can a mining lease be terminated?", "Tell me about mining lease termination", "Mining lease termination conditions"],
#         "responses": ["The State Government can terminate a mining lease if there's a breach or if the lessee transfers the lease without following proper procedures."]
#     },
#     "transfer_of_mining_lease": {
#         "patterns": ["Explain the process of transferring a mining lease", "How can a mining lease be transferred?", "Mining lease transfer rules"],
#         "responses": ["Transfer of a mining lease is allowed with the previous approval of the State Government."]
#     },
#  "prospecting_license": {
#         "patterns": ["How to obtain a prospecting license?", "Tell me about prospecting license", "Procedure for obtaining a prospecting license", "Prospecting license application"],
#         "responses": ["To obtain a prospecting license, follow the procedure in Chapter IX. Provide specific details for a tailored response."]
#     },
#     "mining_lease": {
#         "patterns": ["What are the conditions for a mining lease?", "Tell me about mining lease", "Mining lease application process", "Conditions of mining lease"],
#         "responses": ["Conditions for a mining lease are detailed in Rule 29. For more information, ask a specific question."]
#     },
#     "transfer_of_license": {
#         "patterns": ["Can I transfer a prospecting license?", "Procedure for transferring mining lease", "Transfer of license rules"],
#         "responses": ["Yes, you can transfer a prospecting license. The process is in Rule 23. Specify your question for more details."]
#     },
#     "working_of_mines": {
#         "patterns": ["Prohibition of working of mines", "Can I work on my mines?", "State Government's role in mine working"],
#         "responses": ["The State Government may prohibit mining if there's a contravention. See Rule 32 for details."]
#     },
#     "returns_statements": {
#         "patterns": ["What returns and statements are required?", "Filing returns for mining lease", "Statement submission for prospecting license"],
#         "responses": ["Holders must provide returns as specified in Rule 33. Specify for more detailed information."]
#     },
#     "penalty": {
#         "patterns": ["Penalty for contravention of mining rules", "Consequences of violating mining regulations", "What happens if rules are not followed?"],
#         "responses": ["Violations lead to penalties under Rule 34. For specifics, specify your question."]
#     },
# "revision_application": {
#         "patterns": [
#             "How to apply for revision?",
#             "Revision application process",
#             "Central Government revision application",
#             "Applying for order revision"
#         ],
#         "responses": [
#             "To apply for revision, submit an application to the Central Government within three months of the order. See Rule 35 for details."
#         ]
#     },
#     "application_fee": {
#         "patterns": [
#             "What is the application fee for revision?",
#             "Revision application fee",
#             "Central Government application fee",
#             "How much to pay for revision application?"
#         ],
#         "responses": [
#             "The application fee for revision is a bank draft of rupees ten thousand. Details are in Rule 35(2)."
#         ]
#     },
#     "application_timeframe": {
#         "patterns": [
#             "How long do I have to apply for revision?",
#             "Revision application timeframe",
#             "Deadline for revision application",
#             "When to submit a revision application?"
#         ],
#         "responses": [
#             "You have three months from the date of order communication to apply for revision. Refer to Rule 35(1) for more information."
#         ]
#     },
#     "impleaded_parties": {
#         "patterns": [
#             "Who are impleaded parties in a revision application?",
#             "Implication of parties in revision",
#             "Parties involved in a revision application",
#             "Revision application participants"
#         ],
#         "responses": [
#             "In every application under Rule 35(1), parties to whom a mineral concession was granted for the same area shall be impleaded."
#         ]
#     },
#     "comments_submission": {
#         "patterns": [
#             "How to submit comments for a revision application?",
#             "Providing feedback on revision application",
#             "Comments on Rule 35 application",
#             "Submitting feedback on revision order"
#         ],
#         "responses": [
#             "Comments on a revision application must be submitted within three months from the date of communication. See Rule 35(5) for details."
#         ]
#     },
#     "order_decision": {
#         "patterns": [
#             "How does the Central Government decide on a revision application?",
#             "Central Government's role in revision orders",
#             "Decision-making in revision applications",
#             "Outcome of a revision application"
#         ],
#         "responses": [
#             "The Central Government may confirm, modify, or set aside the order after considering comments. Rule 35(4) provides details."
#         ]
#     },
#     "stay_execution": {
#         "patterns": [
#             "Can the execution of an order be stayed during a revision application?",
#             "Order execution stay in revision application",
#             "Stay of execution in Rule 35 revision",
#             "Central Government stay during revision"
#         ],
#         "responses": [
#             "Yes, the Central Government may stay the execution of the order for sufficient cause during the revision process. See Rule 35(5) for more information."
#         ]
#     },
#  "associated_minerals": {
#         "patterns": [
#             "Which minerals are associated?",
#             "List of associated minerals",
#             "Group of minerals in Section 6",
#             "Tell me about associated minerals"
#         ],
#         "responses": [
#             "The associated minerals for the purposes of Section 6 are categorized as follows:\n(a) Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, Lepidolite, Pitchblende, Samarskite, Scheelite, Topaz, Tantalite, Tourmaline.\n(b) Iron, Manganese, Titanium, Vanadium, and Nickel minerals.\n(c) Lead, Zinc, Copper, Cadmium, Arsenic, Antimony, Bismuth, Cobalt, Nickel, Molybdenum, Uranium minerals, Gold, Silver, Arsenopyrite, Chalcopyrite, Pyrite, Pyrrhotite, and Pentlandite.\n(d) Chromium, Osmiridium, Platinum, and Nickel minerals.\n(e) Kyanite, Sillimanite, Corundum, Dumortierite, and Topaz.\n(f) Gold, Silver, Tellurium, Selenium, and Pyrite.\n(g) Fluorite, Chalcocite, Selenium, and minerals of Zinc, Lead, and Silver.\n(h) Tin and Tungsten minerals.\n(i) Limestone and Magnesite.\n(j) Ilmenite, Monazite, Zircon, Rutile, Leucoxene, Garnet, and Sillimanite.\n(k) Sulphides of Copper and Iron.\n(l) Magnetite and Apatite.\n(m) Magnesite and Chromite.\n(n) Celestite and Phosphatic Nodules.",
#             "These associated minerals play a significant role in various geological and industrial processes. If you have specific questions about any of these minerals or their applications, feel free to ask!"
#         ]
#     },
#  "minerals_valuation": {
#         "patterns": [
#             "What is sale value?",
#             "Explain sale value",
#             "Tell me about mineral valuation",
#             "How is royalty calculated?"
#         ],
#         "responses": [
#             "Sale value is the gross amount payable by the purchaser as indicated in the sale invoice in arms-length transactions. It excludes taxes and no deduction is made for royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust.",
#             "Royalty is chargeable on processed minerals within the leased area or unprocessed run-of-mine removed to a processing plant outside the leased area. The Act specifies royalty calculation based on ad valorem or London Metal Exchange prices, and in some cases, tonnage basis.",
#             "Provisional assessment and adjustment allow lessees to calculate royalty, payments to District Mineral Foundation, and payments to National Mineral Exploration Trust based on the latest available average sale price. Adjustments are made after the publication of average sale prices by the Indian Bureau of Mines.",
#             "The royalty chargeable on a dry basis for metallic ores is based on the metal contained in the ore, calculated using London Metal Exchange or London Bullion Market Association prices. The average sale price of mineral grade/concentrate is computed using the ex-mine price, and the Indian Bureau of Mines publishes this data monthly."
#         ]
#     },
#   "payments": {
#         "patterns": [
#             "How are fees and deposits made?",
#             "Explain payment procedure",
#             "Tell me about making payments",
#             "What is the payment process?"
#         ],
#         "responses": [
#             "Any amount payable under the Act or rules, except for the revision petition fee under rule 35(2), should be paid as specified by the State Government.",
#             "The State Government may charge simple interest at the rate of 24% per annum on rent, royalty, or fee, excluding the fee under rule 35(2), from the sixtieth day after the expiry of the payment due date until the payment is made.",
#             "Mining lease or prospecting licence-cum-mining lease holders are required to pay monies to the District Mineral Foundation and the National Mineral Exploration Trust as per Section 9B and Section 9C, respectively. Additionally, payments under Rule 13 of the Mineral (Auction) Rules, 2015, involve paying the applicable amount quoted under Rule 8 on a monthly basis."
#         ]
#     },
#  "minerals_valuation": {
#         "patterns": [
#             "What is sale value?",
#             "Explain sale value",
#             "Tell me about the sale invoice",
#             "How is sale value calculated?"
#         ],
#         "responses": [
#             "Sale value is the gross amount payable by the purchaser as indicated in the sale invoice, excluding taxes, if any. No deduction is made from the gross amount for royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust when computing sale value.",
#             "Royalty is charged based on whether processing of run-of-mine occurs within or outside the leased area. The Act specifies different methods for calculating royalty based on an ad valorem basis or London Metal Exchange or London Bullion Market Association prices.",
#             "Provisional assessment and adjustment of royalty, payments to the District Mineral Foundation, and payments to the National Mineral Exploration Trust are made at the time of removal or consumption of mineral from the mining lease area."
#         ]
#     },
#     "mining_rules_revision": {
#         "patterns": [
#             "How can one apply for revision?",
#             "Explain the revision process",
#             "Tell me about rule 35",
#             "What is the procedure for revision?"
#         ],
#         "responses": [
#             "Any person aggrieved by an order made by the State Government or other authority can apply for revision within three months of the date of communication of the order. The application should be accompanied by a bank draft or bank transfer for the application fee.",
#             "On receipt of the application for revision, copies are sent to the State Government or other authority and all impleaded parties. Comments and counter-comments are sought, and the Central Government may confirm, modify, or set aside the order.",
#             "The Controller General of Indian Bureau of Mines has the power to issue necessary directions to give effect to the provisions of this chapter."
#         ]
#     },
#     "associated_minerals": {
#         "patterns": [
#             "Tell me about associated minerals",
#             "What are the groups of associated minerals?",
#             "Explain Section 6",
#             "Which minerals are associated?"
#         ],
#         "responses": [
#             "Associated minerals are grouped for the purposes of Section 6. The groups include various minerals such as Apatite, Beryl, Cassiterite, Columbite, and more.",
#             "The associated minerals are categorized into groups such as Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, and more.",
#             "The groups of associated minerals are identified for the purposes of Section 6, and they include various minerals like Iron, Manganese, Titanium, Vanadium, Nickel, Lead, Zinc, Copper, and many more.",
#             "For the purposes of Section 6, the associated minerals are classified into groups such as Apatite, Beryl, Cassiterite, Columbite, Emerald, Felspar, Lepidolite, and more."
#         ]
#     },
#     "minerals_chapter": {
#         "patterns": [
#             "Tell me about minerals in a specific chapter",
#             "Explain a particular chapter on minerals",
#             "Details about a mining chapter",
#             "Which chapter covers certain minerals?"
#         ],
#         "responses": [
#             "Chapter IX covers the procedure for obtaining a prospecting license or mining lease for lands where minerals vest exclusively in a person other than the Government.",
#             "Chapter X pertains to revision and provides details on the application process, order considerations, and comments from involved parties.",
#             "Chapter XI focuses on associated minerals, outlining the groups, classifications, and names of minerals falling under Section 6.",
#             "Chapter XII delves into minerals valuation, discussing topics like sale value, payment of royalty, provisional assessment, and the computation of average sale prices."
#         ]
#     },
#   "compensation": {
#         "patterns": [
#             "Tell me about compensation payment",
#             "How is compensation determined?",
#             "Explain compensation for damage",
#             "When is compensation payable?"
#         ],
#         "responses": [
#             "The holder of a mineral concession is liable to pay annual compensation to the occupier of the surface land. The amount is determined by an officer appointed by the State Government and is based on the average annual net income for agricultural land or the average annual letting value for non-agricultural land for the previous three years.",
#             "For agricultural land, annual compensation is based on the average annual net income from the cultivation of similar land for the previous three years. For non-agricultural land, it is based on the average annual letting value of similar land for the previous three years. The compensation must be paid on or before the specified date by the State Government.",
#             "After the cessation of mining activities due to the expiry, lapsing, surrender, or termination of a mineral concession, the State Government assesses the damage to the land caused by reconnaissance, prospecting, or mining operations. The compensation amount is determined by an officer appointed by the State Government within one year from the date of cessation of mining activities.",
#             "The annual compensation, as mentioned in sub-rule (1), is payable on or before the date specified by the State Government."
#         ]
#     },
#     "penalty": {
#         "patterns": [
#             "What is the penalty for rule contravention?",
#             "Explain the penalty under these rules",
#             "Tell me about rule violation consequences",
#             "What happens if someone breaks the rules?"
#         ],
#         "responses": [
#             "Any contravention of these rules is punishable with imprisonment for up to two years or a fine up to rupees five lakhs, or both. In the case of a continuing contravention, there is an additional fine of up to rupees fifty thousand for every day after conviction for the first contravention.",
#             "Violating these rules can lead to imprisonment for a maximum of two years or a fine up to rupees five lakhs, or both. For continuing violations, there is an extra fine of up to rupees fifty thousand for each day after the first conviction.",
#             "Breaking these rules may result in imprisonment for up to two years or a fine up to rupees five lakhs, or both. If the contravention continues, an additional fine of up to rupees fifty thousand is imposed for each day after the first conviction.",
#             "Consequences of breaking these rules include imprisonment for a term not exceeding two years or a fine up to rupees five lakhs, or both. For continuous violations, there is an additional fine of up to rupees fifty thousand for each day after the initial conviction."
#         ]
#     },
#   "repeal_and_saving": {
#         "patterns": [
#             "Explain repeal and saving under Chapter XVI",
#             "Tell me about cessation of Mineral Concession Rules, 1960",
#             "What happens on the commencement of these rules?",
#             "How are references to Mineral Concession Rules, 1960 replaced?"
#         ],
#         "responses": [
#             "On the commencement of these rules, the Mineral Concession Rules, 1960, cease to be in force with respect to minerals for which the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, are applicable. This ceasing is applicable to things done or omitted to be done before such commencement.",
#             "With respect to minerals covered by these rules, any reference to the Mineral Concession Rules, 1960, in rules made under the Act or any other document is deemed to be replaced with Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, to the extent not repugnant to the context.",
#             "The Mineral Concession Rules, 1960, cease to have effect for minerals under the jurisdiction of the Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, upon the commencement of these rules.",
#             "References to the Mineral Concession Rules, 1960, in any rules made under the Act or other documents are considered as replaced by Minerals (Other than Atomic and Hydrocarbons Energy Minerals) Concession Rules, 2015, to the extent not conflicting with the context upon the commencement of these rules."
#         ]
#     },
#     "amalgamation_of_leases": {
#         "patterns": [
#             "Explain amalgamation of leases under Chapter XVII",
#             "How can two or more leases be amalgamated?",
#             "Tell me about the conditions for amalgamation",
#             "What happens to the period of amalgamated leases?"
#         ],
#         "responses": [
#             "In the interest of mineral development and with reasons recorded in writing, the State Government may permit the amalgamation of two or more adjoining leases held by a lessee.",
#             "The State Government has the authority to permit the amalgamation of two or more adjoining leases held by a lessee if it is in the interest of mineral development. The lessee must provide reasons for the amalgamation, recorded in writing.",
#             "For amalgamation of leases, the period of amalgamated leases is co-terminus with the lease whose period will expire first. The State Government permits amalgamation based on reasons recorded in writing and the interest of mineral development.",
#             "Two or more adjoining leases held by a lessee may be amalgamated by the State Government if it is in the interest of mineral development. The period of amalgamated leases aligns with the lease whose period expires first."
#         ]
#     },
#     "extent_of_area_granted": {
#         "patterns": [
#             "What is the extent of the area granted under a mineral concession?",
#             "Explain the scope of the area granted",
#             "Tell me about the non-mineralised area under a mineral concession",
#             "How is the area defined under a mineral concession?"
#         ],
#         "responses": [
#             "The extent of the area granted under a mineral concession includes the non-mineralised area required for all activities falling under the definition of a mine as defined in clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
#             "Under a mineral concession, the extent of the granted area encompasses the non-mineralised area necessary for activities falling under the definition of a mine according to clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
#             "The extent of the area granted under a mineral concession includes the non-mineralised area needed for all activities defined as a mine under clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952).",
#             "The area granted under a mineral concession extends to the non-mineralised area required for activities falling within the definition of a mine as per clause (j) of sub-section (1) of Section 2 of the Mines Act, 1952 (35 of 1952)."
#         ]
#     },
#     "rectify_apparent_mistakes": {
#         "patterns": [
#             "How can apparent mistakes be rectified?",
#             "Tell me about rectification of mistakes",
#             "Explain the correction of errors in orders",
#             "What is the procedure for rectification of mistakes?"
#         ],
#         "responses": [
#             "Any clerical or arithmetical mistake in any order passed by the Government or any other authority or officer under these rules, and any error arising due to accidental slip or omission, may be corrected by the Government, authority, or officer within two years from the date of the order.",
#             "Rectification of any clerical or arithmetical mistake in an order passed by the Government or any other authority or officer under these rules, and correction of errors due to accidental slip or omission, can be done by the Government, authority, or officer within two years from the date of the order.",
#             "Corrections of clerical or arithmetical mistakes in orders passed by the Government or any other authority or officer under these rules and rectification of errors arising due to accidental slip or omission can be carried out by the Government, authority, or officer within two years from the date of the order.",
#             "Within two years from the date of the order, the Government, authority, or officer may rectify any clerical or arithmetical mistake in an order or correct errors arising due to accidental slip or omission under these rules."
#         ]
#     },
#     "copies_of_licences_and_leases": {
#         "patterns": [
#             "What information must be supplied to the Government?",
#             "Explain the supply of copies of licences and leases",
#             "Tell me about the annual return to be supplied",
#             "How is the information on mineral concessions provided to the Government?"
#         ],
#         "responses": [
#             "A copy of every mineral concession granted or renewed under the Act and rules made thereunder shall be supplied by each State Government within two months of such grant or renewal to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety.",
#             "Every State Government must supply a copy of each mineral concession granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety within two months of the grant or renewal.",
#             "Each State Government is required to supply a consolidated annual return of all mineral concessions granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, not later than the 30th day of June following the year to which the return relates. A copy of this return is also supplied to the Director General, Directorate General of Mines Safety.",
#             "The Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety, must be supplied with a copy of every mineral concession granted or renewed under the Act and rules within two months of such grant or renewal."
#         ]
#     },
#     "copies_of_licences_and_leases": {
#         "patterns": [
#             "What information must be supplied to the Government?",
#             "Explain the supply of copies of licences and leases",
#             "Tell me about the annual return to be supplied",
#             "How is the information on mineral concessions provided to the Government?"
#         ],
#         "responses": [
#             "A copy of every mineral concession granted or renewed under the Act and rules made thereunder shall be supplied by each State Government within two months of such grant or renewal to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety.",
#             "Every State Government must supply a copy of each mineral concession granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety within two months of the grant or renewal.",
#             "Each State Government is required to supply a consolidated annual return of all mineral concessions granted or renewed under the Act and rules made thereunder to the Controller General, Indian Bureau of Mines, not later than the 30th day of June following the year to which the return relates. A copy of this return is also supplied to the Director General, Directorate General of Mines Safety.",
#             "The Controller General, Indian Bureau of Mines, and the Director General, Directorate General of Mines Safety, must be supplied with a copy of every mineral concession granted or renewed under the Act and rules within two months of such grant or renewal."
#         ]
#     }
# }



# training_data = []
# labels = []


# for intent, data in intents.items():
#     if 'patterns' in data:
#         for pattern in data['patterns']:
#             training_data.append(pattern.lower())
#             labels.append(intent)

# vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english", max_df=0.8, min_df=1)
# X_train = vectorizer.fit_transform(training_data)
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, labels, test_size=0.4, random_state=42)

# model = SVC(kernel='linear', probability=True, C=1.0)
# model.fit(X_train, Y_train)




# # Placeholder for your custom model and vectorizer
# def predict_intent(user_input):
#     user_input = user_input.lower()
#     input_vector = vectorizer.transform({user_input})
#     intent = model.predict(input_vector)[0]
#     return intent

# def intentAns(query):
#     user_input = str(query)
#     intent = predict_intent(user_input)
#     try:
#         if intent in intents and 'responses' in intents[intent]:
#             responses = intents[intent]['responses']
#             if responses and len(responses) > 0:
#                 response = random.choice(responses)
#                 print(f"MinesBot: {response}")
#         else:
#             print("MinesBot: I'm sorry, I don't understand your question.")
#     except:
#         # Placeholder for using OpenAI API if your custom model fails
#         client = openai.OpenAIAPI(api_key="your_openai_api_key")
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"{query}"
#                 }
#             ],
#             temperature=0.7,
#             max_tokens=128,
#             top_p=0.9,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         ans = response.choices[0].message.content
#         print(f"MinesBot: {ans}")

# # Your routes
# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input_text = msg
#     return get_Chat_response(input_text)



# def get_Chat_response(text):

#     response = intentAns(text)
#     return response.choices[0].message.content

# if __name__ == '__main__':
#     app.run()
