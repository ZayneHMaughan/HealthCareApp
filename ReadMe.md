The goal of this project is to create a healthcare app whose goal is the following: 
 
$1.$ Connect rural individuals with an app that connects symptoms with a treatment option

$2.$ Passes the individual to a doctor's case-load allowing for prescriptions/complete case management to be done

$3.$ Recommends various other treatments for symptoms and lingering issues that might help


Steps needed to accomplish:

1. Collect the data
   2. I have collected a database but I will see if it works.
   3. issues facing database:
      4. Does not exactly fit this situation it is based on ER visits
      5. Due to this I need to compress the columns and make it more relevant
      6. One idea of columns to get rid of is the _last columns
      7. other colummsn are those with _min and _max _median _count meds_
      8. Columns to keep would be columns that better fit to how a patient is currently feeling
2. Create the database including patients and doctors with different privileges 
3. Create the app background 
   1. If patient logs in, have them input their symptoms. This will be added to a table containing visits
   2. If doctor logs in, they should see a place to manage their cases
