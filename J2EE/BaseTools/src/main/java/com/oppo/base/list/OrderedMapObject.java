/**
 * Copyright (c) 2012 NearMe, All Rights Reserved.
 * FileName:OrderedMapObject.java
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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * ClassName:OrderedMapObject
 * Function: 有序的Map对象
 * @author   80054252
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-16  下午05:22:57
 */
public class OrderedMapObject<K, V> {

	/**
	 * 用于保存数据
	 */
	private Map<String, V> dataMap = new HashMap<String, V>();
	
	/**
	 * 用于保存key值，以支持有序读取
	 */
	private List<String> keyList;
	
	private boolean isForWrite;
	
	/**
	 * 提供两种有序的实现方式
	 *
	 * @param forWrite
	 */
	public OrderedMapObject(boolean isForWrite) {
		this.isForWrite = isForWrite;
		if(isForWrite) {
			// 高效写
			keyList = new LinkedList<String>();
		} else {
			// 高效读取
			keyList = new ArrayList<String>();
		}
	}
	
	/**
	 * 移除所有映射关系
	 * 
	 * @param 
	 * @return
	 */
	public void clear() {
		dataMap.clear();
	}
	
	/**
	 * 是否包含某个key值
	 * 
	 * @param 
	 * @return
	 */
	public boolean containsKey(String key) {
		return dataMap.containsKey(key);
	}
	
	/**
	 * 是否包含某个value值
	 * 
	 * @param 
	 * @return
	 */
	public boolean containsValue(V value) {
		return dataMap.containsValue(value);
	}
	
	/**
	 * 取得某个key对应的值
	 * 
	 * @param 
	 * @return
	 */
	public V get(String key) {
		return dataMap.get(key);
	}
	
	/**
	 * 取得某个位置的值
	 * 
	 * @param 
	 * @return
	 */
	public V get(int index) {
		String key = keyList.get(index);
		return dataMap.get(key);
	}
	
	/**
	 * 验证是否为空
	 * 
	 * @param 
	 * @return
	 */
	public boolean isEmpty() {
		return dataMap.isEmpty();
	}
	
	/**
	 * 存入对象
	 * 
	 * @param 
	 * @return
	 */
	public V put(String key, V value) {
		if(!keyList.contains(key)) {
			keyList.add(key);
		}
		return dataMap.put(key, value);
	}
	
	/**
	 * 存入对象(指定存入位置)
	 * 
	 * @param 
	 * @return
	 */
	public V put(int index, String key, V value) {
		if(keyList.contains(key)) {
			keyList.remove(key);
		}
		keyList.add(index, key);
		
		return dataMap.put(key, value);
	}
	
	/**
	 * 删除指定对象
	 * 
	 * @param 
	 * @return
	 */
	public V remove(String key) {
		keyList.remove(key);
		return dataMap.remove(key);
	}
	
	/**
	 * 删除指定位置的对象
	 * 
	 * @param 
	 * @return
	 */
	public V remove(int index) {
		String key = keyList.get(index);
		keyList.remove(index);
		return dataMap.remove(key);
	}
	
	/**
	 * 取得数据集合的大小
	 * 
	 * @param 
	 * @return
	 */
	public int size() {
		return keyList.size();
	}
	
	/**
	 * 转成List对象
	 * 
	 * @param 
	 * @return
	 */
	public List<V> toList() {
		if(keyList == null || keyList.size() == 0) {
			return null;
		}
		
		List<V> list;
		if(this.isForWrite) {
			list = new LinkedList<V>();
		} else {
			list = new ArrayList<V>();
		}
		
		int size = keyList.size();
		for(int i = 0; i < size; i++) {
			list.add(dataMap.get(keyList.get(i)));
		}
		
		return list;
	}

	/**
	 * 获取isForWrite
	 * @return  isForWrite
	 * @since   Ver 1.0
	 */
	public boolean isForWrite() {
		return isForWrite;
	}
}

