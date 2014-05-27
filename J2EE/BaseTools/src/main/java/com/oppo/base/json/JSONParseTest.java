/**
 * JSONParseTest.java
 * com.oppo.base.json
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-10 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.json;

import junit.framework.TestCase;

/**
 * ClassName:JSONParseTest
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-10  上午09:31:08
 */
public class JSONParseTest extends TestCase {

	protected void setUp() throws Exception {
		super.setUp();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}

	public void testPerson() {
		JSONObjectParser jsonParser = new JSONObjectParser();
		
		String json1 = "{name:'yang',value:'bo'}";
		try {
			Object obj1 = jsonParser.getObjectFromJSON(json1, Person.class);
			System.out.println(obj1);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void testTeam() {
		JSONObjectParser jsonParser = new JSONObjectParser();

		String json2 = "{id:'1',persons:[{name:'yang0',value:'bo0'},{name:'yang1',value:'bo1'}]}";
		try {
			Object obj2 = jsonParser.getObjectFromJSON(json2, Team.class, Person.class);
			System.out.println(obj2);
			System.out.println(jsonParser.generateObjectJSON(obj2, false));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}


