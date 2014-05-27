/**
 * YBTest.java
 * org.json
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-9 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package org.json;

import java.util.HashMap;

/**
 * ClassName:YBTest
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-9  上午09:59:29
 */
public class YBTest {
	public static void main(String[] args) throws JSONException {
		JSONObject json = new JSONObject("{msg:'as',msg1:[{'a':'b','b':'c'},{'a':'d','b':'e'}]}");
		System.out.println(json.getJSONArray("msg")); 
	}
}

