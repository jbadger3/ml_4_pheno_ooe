import os, argparse, sqlite3, json
from collections import OrderedDict
import pandas as pd
import numpy as np

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def construct_demogrpahic_features(example,pt_features_disc,pt_features_cont):
    person_id = example.person_id
    demo_df = pd.read_sql('SELECT * from person WHERE person_id = {}'.format(person_id),connection)
    #each person can have multiple entries in the person table based on the provider and location id
    #just use the first row to collect common information
    demo_df.columns = demo_df.columns.str.lower()
    demos = demo_df.mode().iloc[0]
    dob = demos.birth_datetime
    dob = pd.to_datetime(dob)
    event_date = example.dx_date - offset
    age_at_event = int((event_date - dob).days/365.25)
    age_greater_than_65 = age_at_event > 65.0
    age_less_than_25 = age_at_event < 25.0
    gender = int(demos.gender_concept_id) != 8532 #8532 is concept_id for female gender_code

    #add demographic featrues
    pt_features_disc['person_gender'] = gender
    pt_features_cont['event_person_age'] = age_at_event
    pt_features_disc['lt25_person_age'] = age_less_than_25
    pt_features_disc['gt65_person_age'] = age_greater_than_65

def construct_condition_occurrence_features(example, pt_features_disc,pt_features_cont):
    person_id = example.person_id
    start_date = example.start_date
    end_date = example.end_date
    event_date = example.dx_date - offset
    # person_id, condition_start_date, condition_end_date, condition_concept_id
    query = 'SELECT *  FROM condition_occurrence  WHERE person_id = {} AND condition_start_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    results_df['condition_start_date'] = pd.to_datetime(results_df['condition_start_date'])
    results_df.sort_values('condition_concept_id',inplace=True)
    before_df = results_df[results_df['condition_start_date']< event_date]
    before_counts = before_df['condition_concept_id'].value_counts()
    for concept_id, count in  before_counts.iteritems():
        pt_features_disc['cnt-b_CONCEPT_{}'.format(str(concept_id))] = count

    after_df = results_df[results_df['condition_start_date']>= event_date]
    after_counts = after_df['condition_concept_id'].value_counts()
    for concept_id, count in after_counts.iteritems():
        pt_features_disc['cnt-s_CONCEPT_{}'.format(str(concept_id))] = count


def construct_measurement_features(example, pt_features_disc,pt_features_cont):
    def measurement_interp(results_df):
        #make interpretations an array of shape (rows,5) for low, normal, high, or abnormal lab values
        interpretations = np.zeros((len(results_df),4))
        for result in results_df.itertuples():
            rindex = result.Index
            value_as_number = result.value_as_number
            interp = result.concept_name

            try:
                if not np.isnan(value_as_number):
                    if interp == 'Low':
                        interpretations[rindex,0] = 1 # low
                    elif interp == 'Normal':
                        interpretations[rindex,1] = 1 # normal
                    elif interp == 'High':
                        interpretations[rindex,2] = 1 #high
                    elif interp == 'Abnormal':
                        interpretations[rindex,3] = 1 #Abnormal
            except:
                placeholder= None
        return interpretations

    person_id = example.person_id
    event_date = example.dx_date - offset
    start_date = example.start_date
    end_date = example.end_date
    query = 'SELECT measurement.*,concept.concept_name FROM measurement INNER JOIN concept ON measurement.value_as_concept_id = concept.concept_id   WHERE person_id = {} AND measurement_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    results_df['measurement_date'] = pd.to_datetime(results_df['measurement_date'])
    interp_array = measurement_interp(results_df)
    results_df['interp_low'] = interp_array[:,0]
    results_df['interp_normal'] = interp_array[:,1]
    results_df['interp_high'] = interp_array[:,2]
    results_df['interp_abnormal'] = interp_array[:,3]

    #filter any NAN
    results_df = results_df[results_df['value_as_number'].notnull()]

    results_df.sort_values('measurement_concept_id',inplace=True)
    measurements_before_df = results_df[results_df['measurement_date']< event_date]
    measurements_before_stats = measurements_before_df.groupby('measurement_concept_id')['interp_low','interp_normal','interp_high','interp_abnormal'].agg(['sum'])

    #discrete features (counts)
    for row in measurements_before_stats.iterrows():
        concept_id = row[0]
        interp_low_cnt = row[1][('interp_low','sum')]
        interp_normal_cnt = row[1][('interp_normal','sum')]
        interp_high_cnt = row[1][('interp_high','sum')]
        interp_pos_cnt = row[1][('interp_abnormal','sum')]

        pt_features_disc['cnt-low-b_CONCEPT_{}'.format(str(concept_id))] = interp_low_cnt
        pt_features_disc['cnt-normal-b_CONCEPT_{}'.format(str(concept_id))] = interp_normal_cnt
        pt_features_disc['cnt-high-b_CONCEPT_{}'.format(str(concept_id))] = interp_high_cnt
        pt_features_disc['cnt-abnormal-b_CONCEPT_{}'.format(str(concept_id))] = interp_pos_cnt

    #continuous features (min,max,mean)
    if len(measurements_before_df) >= 1:
        measurements_before_df['value_as_number'] = measurements_before_df['value_as_number'].astype(float)
        measurements_before_stats_cont = measurements_before_df.groupby('measurement_concept_id')['value_as_number'].agg(['min','max','mean'])
        for row in measurements_before_stats_cont.iterrows():
            concept_id = row[0]
            measurement_min = row[1]['min']
            measurement_max = row[1]['max']
            measurement_mean = row[1]['mean']

            pt_features_cont['min-b_CONCEPT_{}'.format(str(concept_id))] = measurement_min
            pt_features_cont['max-b_CONCEPT_{}'.format(str(concept_id))] = measurement_max
            pt_features_cont['mean-b_CONCEPT_{}'.format(str(concept_id))] = measurement_mean

    measurements_after_df = results_df[results_df['measurement_date']>= event_date]
    measurements_after_stats = measurements_after_df.groupby('measurement_concept_id')['interp_low','interp_normal','interp_high','interp_abnormal'].agg(['sum'])

    for row in measurements_after_stats.iterrows():
        concept_id = row[0]
        #discrete features
        interp_low_cnt = row[1][('interp_low','sum')]
        interp_normal_cnt = row[1][('interp_normal','sum')]
        interp_high_cnt = row[1][('interp_high','sum')]
        interp_pos_cnt = row[1][('interp_abnormal','sum')]

        pt_features_disc['cnt-low-s_CONCEPT_{}'.format(str(concept_id))] = interp_low_cnt
        pt_features_disc['cnt-normal-s_CONCEPT_{}'.format(str(concept_id))] = interp_normal_cnt
        pt_features_disc['cnt-high-s_CONCEPT_{}'.format(str(concept_id))] = interp_high_cnt
        pt_features_disc['cnt-abnormal-s_CONCEPT_{}'.format(str(concept_id))] = interp_pos_cnt
    if len(measurements_after_df) >= 1:
        measurements_after_df['value_as_number'] = measurements_after_df['value_as_number'].astype(float)
        measurements_after_stats_cont = measurements_after_df.groupby('measurement_concept_id')['value_as_number'].agg(['min','max','mean'])

        for row in measurements_after_stats_cont.iterrows():
            component_id = row[0]
            measurement_min = row[1]['min']
            measurement_max = row[1]['max']
            measurement_mean = row[1]['mean']

            pt_features_cont['min-s_CONCEPT_{}'.format(str(component_id))] = measurement_min
            pt_features_cont['max-s_CONCEPT_{}'.format(str(component_id))] = measurement_max
            pt_features_cont['mean-s_CONCEPT_{}'.format(str(component_id))] = measurement_mean

def construct_procedures_features(example, pt_features_disc):
    person_id = example.person_id
    start_date = example.start_date
    end_date = example.end_date
    event_date = example.dx_date - offset
    # person_id, condition_start_date, condition_end_date, condition_concept_id
    query = 'SELECT *  FROM procedure_occurrence  WHERE person_id = {} AND procedure_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    results_df['procedure_date'] = pd.to_datetime(results_df['procedure_date'])
    results_df.sort_values('procedure_concept_id',inplace=True)
    before_df = results_df[results_df['procedure_date']< event_date]
    before_counts = before_df['procedure_concept_id'].value_counts()
    for concept_id, count in  before_counts.iteritems():
        pt_features_disc['cnt-b_CONCEPT_{}'.format(str(concept_id))] = count

    after_df = results_df[results_df['procedure_date']>= event_date]
    after_counts = after_df['procedure_concept_id'].value_counts()
    for concept_id, count in after_counts.iteritems():
        pt_features_disc['cnt-s_CONCEPT_{}'.format(str(concept_id))] = count


def construct_obs_features(example, pt_features_disc,pt_features_cont):
    person_id = example.person_id
    event_date = example.dx_date - offset
    start_date = example.start_date
    end_date = example.end_date
    query = 'SELECT * FROM observation  WHERE person_id = {} AND observation_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    results_df['observation_date'] = pd.to_datetime(results_df['observation_date'])

    obs_before_df = results_df[results_df['observation_date']< event_date]
    obs_before_stats = obs_before_df['observation_concept_id'].value_counts()
    for concept_id, count in obs_before_stats.iteritems():
        pt_features_disc['cnt-b_CONCEPT_{}'.format(str(concept_id))] = count

    obs_after_df = results_df[results_df['observation_date']>= event_date]
    obs_after_stats = obs_after_df['observation_concept_id'].value_counts()
    for concept_id, count in obs_after_stats.iteritems():
        pt_features_disc['cnt-s_CONCEPT_{}'.format(str(concept_id))] = count

def construct_device_exposure_features(example, pt_features_disc,pt_features_cont):
    person_id = example.person_id
    event_date = example.dx_date - offset
    start_date = example.start_date
    end_date = example.end_date
    query = 'SELECT * FROM device_exposure  WHERE person_id = {} AND device_exposure_start_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    if not results_df.empty:
        results_df['device_exposure_start_date'] = pd.to_datetime(results_df['device_exposure_start_date'])
        dev_before_df = results_df[results_df['device_exposure_start_date'] < event_date]
        dev_concepts = dev_before_df['device_concept_id']
        for concept_id in dev_concepts:
            pt_features_disc['bool-b_CONCEPT_{}'.format(str(concept_id))] = 1.0
        dev_after_df = results_df[results_df['device_exposure_start_date'] >= event_date]
        dev_concepts = dev_after_df['device_concept_id']
        for concept_id in dev_concepts:
            pt_features_disc['bool-s_CONCEPT_{}'.format(str(concept_id))] = 1.0

def construct_meds_features(example, pt_features_disc,pt_features_cont):
    def filter_opioids(rx_norm_df):
        #remove all medications except for opioids
        rx_norm_df = rx_norm_df[rx_norm_df['quantity'].notnull()]
        rx_norm_df = rx_norm_df[rx_norm_df['days_supply'].notnull()]
        rx_norm_df = rx_norm_df[rx_norm_df['days_supply'] > 0.0]
        rx_norm_df['concept_code'] = rx_norm_df['concept_code'].astype(str)
        rx_norm_df = rx_norm_df[rx_norm_df['concept_code'].isin(rx_cui_to_mme_dict.keys())]
        return rx_norm_df

    def mme_for_drugs(drugs_df):
        # for multistrength drugs each fill will have a row for each ingredient.  we need to filter out
        # the non-opioid strenghts from the list
        opioid_set = set(['Oxymophone','Morphine','Codeine','Tramadol','Hydromorphone','Buprenorphine','Fentanyl','Propoxyphene','Tapentadol','Dihydrocodeine','Oxycodone','Hydrocodone','Methadone'])
        ingredients_string = ','.join(list(drugs_df['ingredient_concept_id'].astype(str)))
        sql_string = 'SELECT concept_id,concept_name from concept where concept_id in ({})'.format(ingredients_string)
        ingredient_names_df = pd.read_sql(sql_string,connection)
        ingredient_names_df.columns = ingredient_names_df.columns.str.lower()
        ingredient_dict = {str(ingredient.concept_id):ingredient.concept_name for ingredient in ingredient_names_df.itertuples()}
        drugs_df['ingredient_name'] = [ingredient_dict[str(concept_id)] for concept_id in drugs_df['ingredient_concept_id']]
        #now remove non-opioid components from list
        opioid_ingredient_mask = [True if ingredient_name in opioid_set else False for ingredient_name in drugs_df['ingredient_name']]
        drugs_df = drugs_df[opioid_ingredient_mask]

        """This calculation makes the following assumptions about fill history, prescribing, and pt behavior.
        1.  The pt is not on multiple medications with the same name unless they are prescribed on the same date. """
        drugs_df.sort_values('drug_exposure_start_date',inplace=True)
        #get the max last date in the list.  then get all drugs filled within 21 days prior to that date. Assume that sample
        #represents represents the patients opioid regimen
        last_fill_date = drugs_df['drug_exposure_start_date'].max()
        drugs_df = drugs_df[drugs_df['drug_exposure_start_date'] >= last_fill_date - pd.Timedelta('21 days')]
        # add strenght information
        opioid_groups = drugs_df.groupby('ingredient_name')
        total_daily_mme = 0
        drug_name_mme_tuples = []
        for drug_name, rx_fills in opioid_groups:
            rx_fills['denominator_value'] = [fill.denominator_value if pd.notnull(fill.denominator_value) else 1.0 for fill in rx_fills.itertuples()]
            rx_fills['strength'] = [float(fill.amount_value) if pd.notnull(fill.amount_value) else float(fill.numerator_value/fill.denominator_value) for fill in rx_fills.itertuples()]
            #AMOUNT_VALUE is the strength of the opioid component in the drug
            last_fills_by_strength = rx_fills.groupby('strength', as_index=False).last()
            # take the last fill group
            if drug_name == 'Fentanyl':
                # the conversion table expects the strength of fentanyl products to be in mcg, so convert any in milligrams
                # rx_fills['NUMERATOR_UNIT_CONCEPT_ID'] == 8576 is mgs
                last_fills_by_strength['strength'] = [fill.strength if int(fill.numerator_unit_concept_id)!=8576 else fill.strength*1000 for fill in last_fills_by_strength.itertuples()]
            last_fills_by_strength['mme'] = [row.quantity/row.days_supply*row.strength*rx_cui_to_mme_dict[str(row.concept_code)] for row in last_fills_by_strength.itertuples()]
            drug_sum = last_fills_by_strength['mme'].sum()
            med_id = last_fills_by_strength.iloc[len(last_fills_by_strength)-1].ingredient_concept_id
            drug_name_mme_tuples.append((drug_name.split(' ')[0],drug_sum,med_id))
        return drug_name_mme_tuples

    person_id = example.person_id
    event_date = example.dx_date - offset
    start_date = example.start_date
    end_date = example.end_date
    cut_off_date = event_date - offset
    query = 'SELECT drug_exposure.*,drug_strength.*, concept.concept_name,concept.vocabulary_id, concept.concept_code FROM drug_exposure INNER JOIN concept ON drug_exposure.drug_concept_id=concept.concept_id INNER JOIN drug_strength ON drug_exposure.drug_concept_id = drug_strength.drug_concept_id  WHERE person_id = {} AND drug_exposure_start_date BETWEEN \'{}\' AND \'{}\';'.format(person_id, start_date, end_date)
    results_df = pd.read_sql_query(query,connection)
    results_df.columns = results_df.columns.str.lower()
    results_df['drug_exposure_start_date'] = pd.to_datetime(results_df['drug_exposure_start_date'])
    #the results can conctain results from HCPCS(for injections), RxNorm, or None if not identifiable.  Select all
    #RxNorm terms first
    rx_norm_df = results_df[results_df['vocabulary_id']=='RxNorm']
    opioids_df = filter_opioids(rx_norm_df)

    #check for dispensed data
    #the drug_type_concept_id field in drug_exposure indicates the type of drug exposure
    #here we gather: 38000175 - prescription dispensed in pharmacy,
    # and 38000176 - prescription dispensed through mail order
    drug_type_concept_ids = [38000175, 38000176]
    opioids_df['drug_type_concept_id'] = opioids_df['drug_type_concept_id'].fillna(0.0)
    opioids_df['drug_type_concept_id'] = opioids_df['drug_type_concept_id'].astype(int)
    is_drug_claim_source_mask = [drug_type in drug_type_concept_ids for drug_type in opioids_df['drug_type_concept_id']]

    if sum(is_drug_claim_source_mask) > 0:
        opioids_df = opioids_df.iloc[is_drug_claim_source_mask]

    opioids_before_df = opioids_df[opioids_df['drug_exposure_start_date'] < cut_off_date]

    if not opioids_before_df.empty:
        drug_name_mme_tuples_b = mme_for_drugs(opioids_before_df)
    else:
        drug_name_mme_tuples_b = []
    mme_b_sum = 0.0
    for drug_name, mme,med_id in drug_name_mme_tuples_b:
        pt_features_disc['has-drug-b_med_{}'.format(med_id)] = 1.0
        pt_features_cont['drug-mme-b_med_{}'.format(med_id)] = mme
        mme_b_sum += mme
    if len(drug_name_mme_tuples_b) > 0:
        if mme_b_sum >= 90.0:
            pt_features_disc['tot-mme-gt-90-b_med_{}'.format(med_id)] = 1.0
        pt_features_cont['tot-mme-b_med_{}'.format(med_id)] = mme_b_sum


    immediately_prior_mask = [True if rx_date >= cut_off_date and rx_date < (event_date + offset) else False for rx_date in opioids_df['drug_exposure_start_date']]
    opioids_immediately_prior_df = opioids_df[immediately_prior_mask]
    if not opioids_immediately_prior_df.empty:
        drug_name_mme_tuples_a = mme_for_drugs(opioids_immediately_prior_df)
    else:
        drug_name_mme_tuples_a = []
    mme_a_sum = 0.0
    for drug_name, mme,med_id in drug_name_mme_tuples_a:
        pt_features_disc['has-drug-s_med_{}'.format(med_id)] = 1.0
        pt_features_cont['drug-mme-s_med_{}'.format(med_id)] = mme
        mme_a_sum += mme
    if len(drug_name_mme_tuples_a) > 0:
        if mme_a_sum >= 90.0:
            pt_features_disc['tot-mme-gte-90-s_med_{}'.format(med_id)] = 1.0
        pt_features_cont['tot-mme-s_med_{}'.format(med_id)] = mme_a_sum



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, help="relative path/to/omop.sqlite database")
    parser.add_argument("--examples_path", type=str, help="relative path/to/examples.csv")
    parser.add_argument("--model_name", type=str, help="model_name")

    args = parser.parse_args()
    assert args.db_path, 'Please use --db_path to specify a path to an OMOP CDM formated sqlite database'
    assert args.examples_path, 'Please use --examples_path to specify a .csv file containing person_ids and labels'
    assert args.model_name, 'Please use --model_name and specify base_model_name'

    #data surrounding each overdose event is divided into two time periods, before and surrounding.
    #The before time period (b) represents 90 days of data collected prior to the overdose (offset by 30 days)
    #The surroudning time period(s) represents 44 days of data tha surround each overdose event (30 days before, 14 days after)
    offset = pd.Timedelta('30 days')
    days_before = pd.Timedelta('90 days')
    days_after = pd.Timedelta('14 days')

    model_name = args.model_name
    output_dir = 'OMOP_features'
    make_dir_if_not_exists(output_dir)

    #open connection to sqlite database
    connection = sqlite3.connect(args.db_path)
    cursor = connection.cursor()

    #load exmaples
    examples_df = pd.read_csv(args.examples_path,parse_dates=['dx_date'])

    #add start and end dates for data collection for each example
    examples_df['start_date'] = [(event_date - offset - days_before).strftime('%Y-%m-%d') for event_date in examples_df['dx_date']]
    examples_df['end_date'] =  [(event_date + days_after).strftime('%Y-%m-%d') for event_date in examples_df['dx_date']]

    #load dictionary of MME conversion data(takes concept_ids for keys)
    with open('supporting_files/mme_OMOP.json','r') as fh:
        rx_cui_to_mme_dict = json.load(fh)

    features_disc_df = pd.DataFrame()
    features_cont_df = pd.DataFrame()
    for example in examples_df.itertuples():
        pt_features_disc = OrderedDict()
        pt_features_cont = OrderedDict()
        construct_demogrpahic_features(example, pt_features_disc,pt_features_cont)
        construct_condition_occurrence_features(example, pt_features_disc,pt_features_cont)
        construct_measurement_features(example, pt_features_disc,pt_features_cont)
        construct_procedures_features(example,pt_features_disc)
        construct_obs_features(example, pt_features_disc,pt_features_cont)
        construct_device_exposure_features(example, pt_features_disc,pt_features_cont)
        construct_meds_features(example, pt_features_disc,pt_features_cont)

        features_disc_df = features_disc_df.append(pt_features_disc,ignore_index=True)
        features_cont_df = features_cont_df.append(pt_features_cont,ignore_index=True)

    features_disc_df.to_csv(os.path.join(output_dir,'{}_features_disc.csv'.format(model_name)),index=False)
    features_cont_df.to_csv(os.path.join(output_dir,'{}_features_cont.csv'.format(model_name)),index=False)
