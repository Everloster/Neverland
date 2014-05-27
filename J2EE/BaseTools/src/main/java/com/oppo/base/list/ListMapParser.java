/**
 * Copyright (c) 2012 NearMe, All Rights Reserved.
 * FileName:ListMapParser.java
 * ProjectName:OBaseTool
 * PackageName:com.oppo.base.list
 * Create Date:2012-1-16
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2012-1-16	  80054252		
 *
 * 
*/

package com.oppo.base.list;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * ClassName:ListMapParser
 * Function: 将List对象转换成Map对象
 * @author   80054252
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-16  下午04:37:04
 */
public class ListMapParser {

	/**
	 * 将List对象以Map形式存储（无序）
	 * 
	 * @param list 数据源
	 * @return
	 */
	public static <T> Map<String, T> getMapFromList(List<T> list, IListHandler<T> handler) {
		
		// 验证list对象是否为空
		if(list == null || list.size() == 0) {
			return null;
		}
		
		Map<String, T> map = new HashMap<String, T>();
		int size = list.size();
		String key;
		
		//逐一取出list中对象放入Map对象中
		for(int i = 0; i < size; i++) {
			T obj = list.get(i);
			key = handler.getMapKey(obj);
			map.put(key, obj);
		}
		
		return map;
	}
	
	/**
	 * 将List对象以Map形式存储（有序）
	 * 
	 * @param list 数据源
	 * @param isForWrite 生成的Map对象是否主要用于写  false--用于读  true--用于写
	 * 		           无论true或false，都可用于读写，只是效率不一样，根据需求选择合适的方式
	 * @return
	 */
	public static <T> OrderedMapObject<String, T> getMapFromListWithOrder(List<T> list, boolean isForWrite, IListHandler<T> handler) {
		
		// 验证list对象是否为空
		if(list == null || list.size() == 0) {
			return null;
		}
		
		OrderedMapObject<String, T> map = new OrderedMapObject<String, T>(isForWrite);
		int size = list.size();
		String key;
		
		//逐一取出list中对象放入Map对象中
		for(int i = 0; i < size; i++) {
			T obj = list.get(i);
			key = handler.getMapKey(obj);
			map.put(key, obj);
		}
		
		return map;
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		List<TestUser> userList = new ArrayList<TestUser>();
		for(int i = 0; i < 10; i++) {
			userList.add(new TestUser("user_" + i, "name_" + i));
		}
		
		Map<String, TestUser> map = getMapFromList(userList, new IListHandler<TestUser>() {
			public String getMapKey(TestUser obj) {
				return obj.getUserId();
			}
		});
		
		OrderedMapObject<String, TestUser> orderedMap = getMapFromListWithOrder(userList, false, new IListHandler<TestUser>() {
			public String getMapKey(TestUser obj) {
				return obj.getUserId();
			}
		});
		
		Set<?> set = map.entrySet();
		Iterator<?> iterator = set.iterator();
		Entry<String, TestUser> entry;
		TestUser user;
		int j = 1;
		while(iterator.hasNext()) {
			entry = (Entry<String, TestUser>)iterator.next();
			user = (TestUser) entry.getValue();
			System.out.println("user:" + j);
			j++;
			System.out.println("user:" + user.getUserId() + "-----" + user.getUserName());
		}
		
		int size = orderedMap.size();
		for(int i = 0; i < size; i++) {
			user = orderedMap.get(i);
			System.out.println("user:" + i);
			System.out.println("user:" + user.getUserId() + "-----" + user.getUserName());
		}
		
		List<TestUser> list = orderedMap.toList();
		System.out.println(list.size());
	}
}

class TestUser {
	private String userId;
	private String userName;
	
	public TestUser(String userId, String userName) {
		this.userId = userId;
		this.userName = userName;
	}
	
	/**
	 * 获取userId
	 * @return  userId
	 * @since   Ver 1.0
	 */
	public String getUserId() {
		return userId;
	}
	/**
	 * 设置userId  
	 * @param   userId	.   
	 * @since   Ver 1.0
	 */
	public void setUserId(String userId) {
		this.userId = userId;
	}
	/**
	 * 获取userName
	 * @return  userName
	 * @since   Ver 1.0
	 */
	public String getUserName() {
		return userName;
	}
	/**
	 * 设置userName  
	 * @param   userName	.   
	 * @since   Ver 1.0
	 */
	public void setUserName(String userName) {
		this.userName = userName;
	}
}

