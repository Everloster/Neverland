/**
 * JSONParser.java
 * com.oppo.base.json
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-9 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.json;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;
import org.w3c.dom.Node;

import com.oppo.base.beanutils.BeanUtil;
import com.oppo.base.beanutils.DefaultNameTransfer;
import com.oppo.base.beanutils.INameTransfer;
import com.oppo.base.beanutils.IValueTransfer;
import com.oppo.base.beanutils.MethodsUtil;
import com.oppo.base.beanutils.ObjectValueTransfer;
import com.oppo.base.beanutils.StringValueTransfer;
import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;
import com.oppo.base.time.TimeFormat;

/**
 * ClassName:JSONParser
 * Function: 将一个独立的xml节点解析为对象的工具
 *
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-8-9 下午03:10:01
 */
public class JSONObjectParser {
	private INameTransfer nameTransfer;
	private StringValueTransfer strValueTransfer;
	private ObjectValueTransfer objValueTransfer;
	private String encoding;

	private final static DefaultNameTransfer DEFAULT_NAME_TRANSFER = new DefaultNameTransfer();

	public JSONObjectParser() {
	}

	/**
	 * 根据指定的类型解析json
	 *
	 * @param json
	 *            json数据
	 * @param tClass
	 *            数据实体类型
	 */
	public <T> T getObjectFromJSON(String json, Class<T> tClass) throws Exception {
		if (StringUtil.isNullOrEmpty(json)) {
			return null;
		}

		return parseJSON(new JSONObject(json), tClass, null);
	}

	/**
	 * 根据指定的类型解析json
	 *
	 * @param json json数据
	 * @param tClass 数据实体类型
	 * @param dClass 数据实体中的list的类型
	 */
	public <T, D> T getObjectFromJSON(String json, Class<T> tClass, Class<D> dClass) throws Exception {
		if (StringUtil.isNullOrEmpty(json)) {
			return null;
		}

		return parseJSON(new JSONObject(json), tClass, dClass);
	}

	/**
	 * 根据指定的类型解析json为list
	 *
	 * @param json json数据
	 * @param tClass 数据实体类型
	 */
	public <T> List<T> getListFromJSON(String json, Class<T> tClass) throws Exception {
		return getListFromJSON(json, tClass, null);
	}

	/**
	 * 根据指定的类型解析json为list
	 *
	 * @param json json数据
	 * @param tClass 数据实体类型
	 */
	public <T, D> List<T> getListFromJSON(String json, Class<T> tClass, Class<D> dClass) throws Exception {
		if (StringUtil.isNullOrEmpty(json)) {
			return null;
		}

		//解析得到json数组
		JSONArray array = new JSONArray(json);
		int length = array.length();
		if (length == 0) {
			return null;
		}

		//逐个解析数组中的数据
		List<T> tList = new ArrayList<T>(length);
		for (int i = 0; i < length; i++) {
			T t = parseJSON(array.getJSONObject(i), tClass, dClass);
			if (null != t) {
				tList.add(t);
			}
		}

		return tList;
	}

	/**
	 * 将节点中的子节点解析为对象属性
	 *
	 * @param <T> 实体类型
	 * @param node Xml节点
	 * @param tClass
	 * @param nameTrans xml节点与实体属性名称转换器
	 * @return
	 * @throws Exception
	 */
	public <T, D> T parseJSON(JSONObject jsonObject, Class<T> tClass,
			Class<D> dClass) throws Exception {
		if (null == jsonObject) {
			return null;
		}

		T tObj = tClass.newInstance();// 创建实体
		boolean isSetParam = false;

		Iterator<?> it = jsonObject.keys();
		while (it.hasNext()) {
			String key = it.next().toString();
			Object value = jsonObject.get(key);
			if (value instanceof JSONArray) {
				boolean isNodeSet = handleArrayProperty(key, (JSONArray) value,
						tObj, dClass);
				if (isNodeSet) {
					isSetParam = true;
				}
			} else {// 单个属性
				boolean isTextSet = handleSingleProperty(key, value.toString(),
						tObj);
				if (isTextSet) {
					isSetParam = true;
				}
			}
		}

		return isSetParam ? tObj : null;
	}

	/**
	 * 处理text类型的子节点
	 *
	 * @param
	 * @return 该节点是否有效
	 */
	protected <T> boolean handleSingleProperty(String name, String value, T tObj)
			throws Exception {
		if (!StringUtil.isNullOrEmpty(value)) {// 如果非空
			return BeanUtil.setValueToObject(getNameTransfer(),
					getStrValueTransfer(), tObj,
					MethodsUtil.getMethods(tObj.getClass()), name, value);
		}

		return false;
	}

	/**
	 * 处理非text类型的子节点
	 *
	 * @param
	 * @return
	 */
	protected <T, D> boolean handleArrayProperty(String name,
			JSONArray jsonArray, T tObj, Class<D> dClass) throws Exception {
		String methodName = getNameTransfer().getGetMethod(name);
		Method m = MethodsUtil.getMethod(tObj.getClass(), methodName);
		if (List.class.isAssignableFrom(m.getReturnType())) {
			int len = jsonArray.length();
			List<D> propList = new ArrayList<D>(len);
			for (int i = 0; i < len; i++) {
				JSONObject jsonObj = (JSONObject) jsonArray.get(i);
				D dObj = this.parseJSON(jsonObj, dClass, null);
				if (null != dObj) {
					propList.add(dObj);
				}
			}

			return BeanUtil.setValueToObject(getNameTransfer(),
					getObjValueTransfer(), tObj,
					MethodsUtil.getMethods(tObj.getClass()), name, propList);
		} else {
			// 空集合
			if(jsonArray.length() == 0) {
				return false;
			}

			// 非list当做String
			return BeanUtil.setValueToObject(getNameTransfer(),
						getObjValueTransfer(), tObj,
						MethodsUtil.getMethods(tObj.getClass()), name, jsonArray.toString());
		}
	}

	/**
	 * 产生JSON对象的字符串形式
	 *
	 * @param
	 * @return
	 */
	public String generateObjectJSON(Object obj) throws Exception {
		return generateObjectJSON(obj, false, null, null);
	}

	/**
	 * 产生对象的JSON字符串形式
	 *
	 * @param obj 被转换的对象
	 * @param needEmpty 是否需要得到值为空的属性
	 * @return
	 */
	public String generateObjectJSON(Object obj, boolean needEmpty)
			throws Exception {
		return generateObjectJSON(obj, needEmpty, null, null);
	}

	/**
	 * 产生对象的JSON字符串形式
	 *
	 * @param obj 被转换的对象
	 * @param needEmpty 是否需要得到值为空的属性
	 * @param includeMethod 只解析对应的字段，如果此属性存在值则忽略excludeMethod
	 * @param excludeMethod 解析时去掉此属性中指定的字段
	 * @return
	 */
	public String generateObjectJSON(Object obj, String[] includeMethods, String[] excludeMethods)
			throws Exception{
		return generateObjectJSON(obj, false, includeMethods, excludeMethods);
	}

	/**
	 * 产生对象的JSON字符串形式
	 *
	 * @param obj 被转换的对象
	 * @param needEmpty 是否需要得到值为空的属性
	 * @param includeMethod 只解析对应的字段，如果此属性存在值则忽略excludeMethod
	 * @param excludeMethod 解析时去掉此属性中指定的字段
	 * @return
	 */
	public String generateObjectJSON(Object obj, boolean needEmpty, String[] includeMethods, String[] excludeMethods)
			throws Exception{
		if (null == obj) {
			return OConstants.EMPTY_STRING;
		}

		if (obj instanceof Iterable) {
			JSONArray jsonArray = this.getIterableValue((Iterable<?>)obj, needEmpty, includeMethods, excludeMethods);
			return jsonArray.toString();
		} else {
			JSONObject jsonObject = generateObjectJSONObject(obj, needEmpty, includeMethods, excludeMethods);
			return jsonObject.toString();
		}
	}

	private JSONObject generateObjectJSONObject(Object obj, boolean needEmpty, String[] includeMethods, String[] excludeMethods)
			throws Exception {
		JSONObject jsonObject = new JSONObject();

		// 逐个解析obj中的方法
		Method[] ms = MethodsUtil.getMethods(obj.getClass());
		int mLen = ms.length;

		for (int i = 0; i < mLen; i++) {
			Method method = ms[i];
			if (!Modifier.isPublic(method.getModifiers())) {
				continue;
			}

			String methodName = method.getName();
			if (methodName.startsWith("get") && !methodName.equals("getClass")) {
				// 节点名
				String nodeName = getNameTransfer().transferToNodeName(
						methodName);

				//判断是否可以被解析
				if(!canParse(nodeName, includeMethods, excludeMethods)) {
					continue;
				}

				// 获取属性值
				Object value = method.invoke(obj);
				// 转换为字符串
				Object propertyValue = getPropertyValue(value, needEmpty, includeMethods, excludeMethods);

				if (needEmpty && null == propertyValue) {
					propertyValue = "";
				}

				if (null != propertyValue) {
					jsonObject.put(nodeName, propertyValue);
				}
			}
		}

		return jsonObject;
	}

	/**
	 * 判断指定名称在规则下是否可以进行解析
	 * @param
	 * @return
	 */
	protected boolean canParse(String name, String[] includeMethods, String[] excludeMethods) {
		if(null != includeMethods) {
			//优先包含的
			for(String includeMethod : includeMethods) {
				if(name.equals(includeMethod)) {
					return true;
				}
			}

			return false;
		} else if(null != excludeMethods) {
			for(String excludeMethod : excludeMethods) {
				if(name.equals(excludeMethod)) {
					return false;
				}
			}

			return true;
		}

		return true;
	}

	/**
	 * 检查根节点是否合法
	 *
	 * @param node 根节点
	 * @return
	 */
	protected boolean checkRootNode(Node node) {
		return null != node;
	}

	/**
	 * 获取属性值的toString()值
	 *
	 * @param propertyValue 属性值
	 * @param needEmpty 是否需要空属性
	 * @return
	 */
	protected Object getPropertyValue(Object propertyValue, boolean needEmpty, String[] includeMethod, String[] excludeMethod)
			throws Exception {
		if ((propertyValue instanceof Iterable)) {
			return getIterableValue((Iterable<?>) propertyValue, needEmpty, includeMethod, excludeMethod);
		} else if (propertyValue instanceof Map) {
			return null;
		}

		String strValue = "";
		if (null != propertyValue) {
			// 如果是Date类型
			if (propertyValue instanceof Date) {
				strValue = TimeFormat.getTimeString((Date) propertyValue);
			} else if (propertyValue.getClass().isSynthetic()) {
				return generateObjectJSONObject(propertyValue, needEmpty, includeMethod, excludeMethod);
			} else if (propertyValue instanceof Integer) {
				if (0 == (Integer)propertyValue) {
					return null;
				}

				strValue = propertyValue.toString();
			} else if (propertyValue instanceof Long) {
				if (0 == (Long)propertyValue) {
					return null;
				}

				strValue = propertyValue.toString();
			} else if (propertyValue instanceof Double) {
				if (0 == (Double)propertyValue) {
					return null;
				}

				strValue = propertyValue.toString();
			} else if (propertyValue instanceof Boolean) {
				if (!(Boolean)propertyValue) {
					return null;
				}

				strValue = propertyValue.toString();
			} else if (propertyValue instanceof Float) {
				if (0 == (Float)propertyValue) {
					return null;
				}

				strValue = propertyValue.toString();
			} else {
				// 尝试转换为json格式
				strValue = propertyValue.toString();
			}
		}

		if ((!needEmpty && strValue.length() > 0) || needEmpty) {
			return strValue;
		}

		return null;
	}

	protected JSONArray getIterableValue(Iterable<?> iterable, boolean needEmpty, String[] includeMethod, String[] excludeMethod)
			throws Exception {
		JSONArray jsonArray = new JSONArray();
		Iterator<?> it = iterable.iterator();
		while (it.hasNext()) {
			Object obj = it.next();

			if(obj instanceof String
					|| obj instanceof Long
					|| obj instanceof Integer
					|| obj instanceof Double
					|| obj instanceof Boolean) {
				jsonArray.put(obj);
			} else {
				jsonArray.put(generateObjectJSONObject(obj, needEmpty, includeMethod, excludeMethod));
			}
		}

		return jsonArray;
	}

	/**
	 * 获取名称转换器
	 *
	 * @return the nameTransfer
	 * @since Ver 1.0
	 */
	public INameTransfer getNameTransfer() {
		if (null == nameTransfer) {
			nameTransfer = DEFAULT_NAME_TRANSFER;
		}

		return nameTransfer;
	}

	/**
	 * 设置名称转换器
	 *
	 * @param nameTransfer
	 * @since Ver 1.0
	 */
	public void setNameTransfer(INameTransfer nameTransfer) {
		this.nameTransfer = nameTransfer;
	}

	/**
	 * 获取值转换器
	 *
	 * @return the valueTransfer
	 * @since Ver 1.0
	 */
	public IValueTransfer<String> getStrValueTransfer() {
		if (null == strValueTransfer) {
			strValueTransfer = new StringValueTransfer();
		}

		return strValueTransfer;
	}

	/**
	 * 设置值转换器
	 *
	 * @param valueTransfer
	 * @since Ver 1.0
	 */
	public void setStrValueTransfer(StringValueTransfer strValueTransfer) {
		this.strValueTransfer = strValueTransfer;
	}

	/**
	 * 获取objValueTransfer
	 *
	 * @return the objValueTransfer
	 * @since Ver 1.0
	 */
	public ObjectValueTransfer getObjValueTransfer() {
		if (null == objValueTransfer) {
			objValueTransfer = new ObjectValueTransfer();
		}
		return objValueTransfer;
	}

	/**
	 * 设置objValueTransfer
	 *
	 * @param objValueTransfer
	 * @since Ver 1.0
	 */
	public void setObjValueTransfer(ObjectValueTransfer objValueTransfer) {
		this.objValueTransfer = objValueTransfer;
	}

	/**
	 * 获取encoding
	 *
	 * @return the encoding
	 * @since Ver 1.0
	 */
	public String getEncoding() {
		if (null == encoding) {
			encoding = OConstants.DEFAULT_ENCODING;
		}

		return encoding;
	}

	/**
	 * 设置encoding
	 *
	 * @param encoding
	 * @since Ver 1.0
	 */
	public void setEncoding(String encoding) {
		this.encoding = encoding;
	}
}
