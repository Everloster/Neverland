/**
 * Team.java
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

import java.util.ArrayList;

/**
 * ClassName:Team
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-10  上午09:36:18
 */
public class Team {
	private int id;
	private ArrayList<Person> persons;
	/**
	 * 获取id
	 * @return  the id
	 * @since   Ver 1.0
	 */
	public int getId() {
		return id;
	}
	/**
	 * 设置id
	 * @param   id    
	 * @since   Ver 1.0
	 */
	public void setId(int id) {
		this.id = id;
	}
	/**
	 * 获取person
	 * @return  the person
	 * @since   Ver 1.0
	 */
	public ArrayList<Person> getPersons() {
		return persons;
	}
	/**
	 * 设置person
	 * @param   person    
	 * @since   Ver 1.0
	 */
	public void setPersons(ArrayList<Person> persons) {
		this.persons = persons;
	}
	
	public String toString() {
		String strPersons = "";
		if(null != persons) {
			for(int i = 0; i < persons.size(); i++) {
				strPersons += persons.get(i).toString()+",";
			}
		}
		return String.format("{id:%d,persons:[%s]}", id , strPersons);
	}
}

