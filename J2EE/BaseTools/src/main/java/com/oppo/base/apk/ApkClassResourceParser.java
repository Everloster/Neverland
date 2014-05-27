/**
 * ApkResourceParser.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-18 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.apk;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jf.dexlib.ClassDataItem;
import org.jf.dexlib.ClassDefItem;
import org.jf.dexlib.DexFile;
import org.jf.dexlib.EncodedArrayItem;
import org.jf.dexlib.EncodedValue.EncodedValue;

import com.oppo.base.apk.entity.ApkResourceProperty;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:ApkResourceParser
 * Function: 解析apk包中的资源类,解析完成后请手动关闭打开的InputStream
 * FileInputStream fis = null;
 * try {
 * 		fis = new FileInputStream("F:\\资料\\java\\apk\\classes.dex");
 * 		ApkClassResourceParser parse = new ApkClassResourceParser();
 * 		parse.parseResource(fis, "com.gker.five");
 * 	} catch(Exception ex) {
 * 		ex.printStackTrace();
 * 	} finally {
 * 		FileOperate.close(fis);
 * 	}
 *
 * @author   80036381
 * @version
 * @see com.oppo.base.apk.entity.ApkResourceProperty
 * @since    Ver 1.1
 * @Date	 2011-7-18  上午08:55:36
 *
 */
public class ApkClassResourceParser {
	/**
	 * 内部类标记
	 */
	private static final String INNER_CLASS_FLAG = "$";

	/**
	 * 资源类名
	 */
	private static final String RESOURCE_CLASS = ".R";

	private Map<String, ApkResourceProperty> resourceMap;

	/**
	 * 搜索指定package name中的指定资源
	 * @param
	 * @return
	 */
	public ApkResourceProperty search(File classFile, String packageName, String address) throws Exception {
		if(StringUtil.isNullOrEmpty(packageName)) {
			throw new NullPointerException("package can not be empty");
		} else if(StringUtil.isNullOrEmpty(address)) {
			throw new NullPointerException("address can not be empty");
		}

		return this.parseResourceForAddress(classFile, packageName, address);
	}

	/**
	 * 解析apk包中的class文件，获取其中的资源路径
	 * @param classStream classes.dex文件对应的流
	 * @param packageName 当前apk对应的包名
	 * @return 资源map
	 */
	public void parseResource(File classFile, String packageName) throws IOException {
		resourceMap = new HashMap<String, ApkResourceProperty>();

		DexFile dexFile = new DexFile(classFile);

		boolean hasPackageName = !StringUtil.isNullOrEmpty(packageName);
		String resourceClassName = null;	//包对应的资源文件名
		if(hasPackageName) {
			resourceClassName = packageName + RESOURCE_CLASS;
		} else {
			resourceClassName = RESOURCE_CLASS;
		}
		String resourceInnerClass = resourceClassName + INNER_CLASS_FLAG;

		ArrayList<ClassDefItem> classDefItems = new ArrayList<ClassDefItem>(dexFile.ClassDefsSection.getItems());
		for (ClassDefItem classDefItem: classDefItems) {
			String classDescriptor = classDefItem.getClassType().getTypeDescriptor();

			int lastIndex = classDescriptor.length() - 1;
            //validate that the descriptor is formatted like we expect
            if (classDescriptor.charAt(0) != 'L' || classDescriptor.charAt(lastIndex) != ';') {
                System.err.println("Unrecognized class descriptor - " + classDescriptor + " - skipping class");
                continue;
            }

            String className = classDescriptor.substring(1, lastIndex).replaceAll("/", ".");

            boolean isResourceClass = false;
            if((hasPackageName && className.startsWith(resourceInnerClass))
            		|| (!hasPackageName && className.indexOf(resourceInnerClass) > -1)) {
            	isResourceClass = true;
            }

            if(isResourceClass) {
            	ClassDataItem classDataItem = classDefItem.getClassData();
            	if(null == classDataItem) {
            		continue;
            	}

            	//处理资源类
                List<ClassDataItem.EncodedField> encodedFields = classDataItem.getStaticFields();
                if (null != encodedFields && encodedFields.size() > 0) {
                	//
                	EncodedArrayItem encodedStaticInitializers = classDefItem.getStaticFieldInitializers();

                    EncodedValue[] staticInitializers;
                    if (encodedStaticInitializers != null) {
                        staticInitializers = encodedStaticInitializers.getEncodedArray().values;
                    } else {
                        staticInitializers = new EncodedValue[0];
                    }
                    //获取资源位置
                    String resourceFolder = className.substring(resourceInnerClass.length());

                	for (int i=0, size = encodedFields.size(); i < size; i++) {
                		ClassDataItem.EncodedField field = encodedFields.get(i);
                		if(!field.isStatic()) {
                			continue;
                		}

                        if (i < staticInitializers.length) {
                        	//获取值
                        	EncodedValue encodedValue = staticInitializers[i];
                        	if(null != encodedValue) {
                        		String value = encodedValue.toString();
                        		int index = value.lastIndexOf('@');
                        		if(index > 0) {
                        			String address = value.substring(index);
                        			//资源位置
                            		String resource = field.field.getFieldName().getStringValue();
                            		resourceMap.put(address, new ApkResourceProperty(resourceFolder,resource, address));
                        		}
                        	}
                        }
                	}

                	staticInitializers = null;
                }
            }
		}
	}

	/**
	 * 解析apk包中的class文件，获取其中的资源路径
	 * @param classStream classes.dex文件对应的流
	 * @param packageName 当前apk对应的包名
	 * @return 资源map
	 */
	public ApkResourceProperty parseResourceForAddress(File classFile, String packageName, String searchAddress) throws IOException {
		DexFile dexFile = new DexFile(classFile);

		boolean hasPackageName = !StringUtil.isNullOrEmpty(packageName);
		String resourceClassName = null;	//包对应的资源文件名
		if(hasPackageName) {
			resourceClassName = packageName + RESOURCE_CLASS;
		} else {
			resourceClassName = RESOURCE_CLASS;
		}
		String resourceInnerClass = resourceClassName + INNER_CLASS_FLAG;

		ArrayList<ClassDefItem> classDefItems = new ArrayList<ClassDefItem>(dexFile.ClassDefsSection.getItems());
		for (ClassDefItem classDefItem: classDefItems) {
			String classDescriptor = classDefItem.getClassType().getTypeDescriptor();

			int lastIndex = classDescriptor.length() - 1;
            //validate that the descriptor is formatted like we expect
            if (classDescriptor.charAt(0) != 'L' || classDescriptor.charAt(lastIndex) != ';') {
                System.err.println("Unrecognized class descriptor - " + classDescriptor + " - skipping class");
                continue;
            }

            String className = classDescriptor.substring(1, lastIndex).replaceAll("/", ".");

            boolean isResourceClass = false;
            if((hasPackageName && className.startsWith(resourceInnerClass))
            		|| (!hasPackageName && className.indexOf(resourceInnerClass) > -1)) {
            	isResourceClass = true;
            }

            if(isResourceClass) {
            	ClassDataItem classDataItem = classDefItem.getClassData();
            	if(null == classDataItem) {
            		continue;
            	}

            	//处理资源类
                List<ClassDataItem.EncodedField> encodedFields = classDataItem.getStaticFields();
                if (null != encodedFields && encodedFields.size() > 0) {
                	//
                	EncodedArrayItem encodedStaticInitializers = classDefItem.getStaticFieldInitializers();

                    EncodedValue[] staticInitializers;
                    if (encodedStaticInitializers != null) {
                        staticInitializers = encodedStaticInitializers.getEncodedArray().values;
                    } else {
                        staticInitializers = new EncodedValue[0];
                    }
                    //获取资源位置
                    String resourceFolder = className.substring(resourceInnerClass.length());

                	for (int i=0, size = encodedFields.size(); i < size; i++) {
                		ClassDataItem.EncodedField field = encodedFields.get(i);
                		if(!field.isStatic()) {
                			continue;
                		}

                        if (i < staticInitializers.length) {
                        	//获取值
                        	EncodedValue encodedValue = staticInitializers[i];
                        	if(null != encodedValue) {
                        		String value = encodedValue.toString();
                        		int index = value.lastIndexOf('@');
                        		if(index > 0) {
                        			String address = value.substring(index);
                        			//资源位置
                            		String resource = field.field.getFieldName().getStringValue();
                            		if(searchAddress.equals(address)) {
                            			return new ApkResourceProperty(resourceFolder,resource, address);
                            		}
                        		}
                        	}
                        }
                	}

                	staticInitializers = null;
                }
            }
		}

		return null;
	}

	/**
	 * 根据地址获取资源的属性
	 * @param
	 * @return
	 */
	public ApkResourceProperty getResource(String address) {
		if(null == resourceMap) {
			return null;
		}

		return resourceMap.get(address.toLowerCase());
	}

	/**
	 * 获取解析到的所有资源信息
	 * @param
	 * @return
	 */
	public Map<String, ApkResourceProperty> getAllResource() {
		return resourceMap;
	}

	/**
	 * Function Description here
	 * @param
	 * @return
	 */
	public static void main(String[] args) {
		try {
			File f = new File("H:\\classes.dex");
			ApkClassResourceParser parse = new ApkClassResourceParser();
			parse.parseResource(f, "com.oppo.LockScreenGlassBoard");
			Map<String, ApkResourceProperty> resourceMap = parse.getAllResource();
			for(String key : resourceMap.keySet()) {
				System.out.println(key + "\t" + resourceMap.get(key));
			}
		} catch(Exception ex) {
			ex.printStackTrace();
		}
	}
}

