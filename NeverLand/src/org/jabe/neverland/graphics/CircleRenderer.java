package org.jabe.neverland.graphics;

import java.io.IOException;
import java.io.InputStream;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import org.jabe.neverland.R;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLSurfaceView.Renderer;
import android.opengl.GLUtils;

public class CircleRenderer implements Renderer {

	//声明并实例化圆形实体
	private Circle circle;
	
//	public float change;
	
	//上下文引用
	private Context context;
	
	/**
	 * 渲染器构造函数
	 */
	public CircleRenderer(Context context){
		this.context = context;
	}
	
	@Override
	public void onDrawFrame(GL10 gl) {
		//启动阴影平滑
		gl.glShadeModel(GL10.GL_SMOOTH);
		//设置开灯效果
		if(DataConfig.openLightFlag){//开灯
			//启用光照
			gl.glEnable(GL10.GL_LIGHTING);
			//初始0号灯
			initLight0(gl);
			//初始化白色材质
			initMaterialWhite(gl);
			//设置0号灯光源的位置
			float[] positionParamsGreen={2,1,0,1};//最后的1表示是定位光
			gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_POSITION, positionParamsGreen,0); 
		}else{//关灯
			gl.glDisable(GL10.GL_LIGHTING);
		}
		//清除颜色和深度缓存
		gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
		//设置当前矩阵为模式矩阵
		gl.glMatrixMode(GL10.GL_MODELVIEW);
		//设置当前矩阵为单位矩阵
		gl.glLoadIdentity();
		//向z轴方向移动
		gl.glTranslatef(0.0f, 0.0f, -2.0f);
		
		//保护变换矩阵现场
		gl.glPushMatrix();
		//圆形绘制
		circle.drawSelf(gl);
		//恢复变换矩阵现场
		gl.glPopMatrix();
	}

	@Override
	public void onSurfaceChanged(GL10 gl, int width, int height) {
		//设置视窗大小及位置
		gl.glViewport(0, 0, width, height);
		//设置当前矩阵为投影矩阵
		gl.glMatrixMode(GL10.GL_PROJECTION);
		//设置当前矩阵为单位矩阵
		gl.glLoadIdentity();
		//计算透视投影比例
		float ratio = (float) width/height;
		//计算产生透视投影矩阵
		gl.glFrustumf(-ratio, ratio, -1, 1, 1, 10);
	}

	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config) {
		//关闭抗抖动
		gl.glDisable(GL10.GL_DITHER);
		//设置特定Hint项目的模式,这是设置为使用快速模式
		gl.glHint(GL10.GL_PERSPECTIVE_CORRECTION_HINT, GL10.GL_FASTEST);
		//设置屏幕背景为黑色RGBA
		gl.glClearColor(0, 0, 0, 0);
		//设置着色模式为平滑着色
		gl.glShadeModel(GL10.GL_SMOOTH);
		//启用深度测试
		gl.glEnable(GL10.GL_DEPTH_TEST);
		
		int textureId = initTexture(gl,R.drawable.jw);
		circle = new Circle(4, textureId);
	}
	
	/**
	 * 初始化0号灯
	 * @param gl
	 */
	private void initLight0(GL10 gl){
		//打开0号灯
		gl.glEnable(GL10.GL_LIGHT0);
		//环境光设置
		float[] ambientParams = {0.1f,0.1f,0.1f,1.0f};
		gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_AMBIENT, ambientParams,0);
		//散射光设置
		float[] diffuseParams = {0.5f,0.5f,0.5f,1.0f};
		gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_DIFFUSE, diffuseParams,0);
		//反射光设置
		float[] specularParams = {1.0f,1.0f,1.0f,1.0f};
		gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_SPECULAR, specularParams,0);
	}
	
	/**
	 * 初始化白色材质
	 * 	材质为白色时什么颜色的光照在上面就将体现出什么颜色
	 * @param gl
	 */
	private void initMaterialWhite(GL10 gl){
		//环境光为白色材质
		float[] ambientMaterial = {0.4f, 0.4f, 0.4f, 1.0f};
		gl.glMaterialfv(GL10.GL_FRONT_AND_BACK, GL10.GL_AMBIENT, ambientMaterial,0);
        //散射光为白色材质
        float[] diffuseMaterial = {0.8f, 0.8f, 0.8f, 1.0f};
        gl.glMaterialfv(GL10.GL_FRONT_AND_BACK, GL10.GL_DIFFUSE, diffuseMaterial,0);
        //高光材质为白色
        float[] specularMaterial = {1.0f, 1.0f, 1.0f, 1.0f};
        gl.glMaterialfv(GL10.GL_FRONT_AND_BACK, GL10.GL_SPECULAR, specularMaterial,0);
        //高光反射区域,数越大高亮区域越小越暗
        float[] shininessMaterial = {1.5f};
        gl.glMaterialfv(GL10.GL_FRONT_AND_BACK, GL10.GL_SHININESS, shininessMaterial,0);
	}
	
	/**
	 * 初始化纹理
	 * @param gl
	 * @param textureId
	 * @return
	 */
	public int initTexture(GL10 gl,int textureId){
		//创建数组
		int[] textures = new int[1];
		//生成纹理数组
		gl.glGenTextures(1, textures,0);
		//赋值
		int currTextureId=textures[0];
		//绑定纹理ID
		gl.glBindTexture(GL10.GL_TEXTURE_2D, currTextureId);
		
		//设置当期纹理映射方式
		//指定缩小过滤方式
		gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_MIN_FILTER,GL10.GL_NEAREST);
		//指定放大过滤方式
	    gl.glTexParameterf(GL10.GL_TEXTURE_2D,GL10.GL_TEXTURE_MAG_FILTER,GL10.GL_LINEAR);
	    //指定S坐标轴贴图模式
	    gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_WRAP_S,GL10.GL_REPEAT);
	    //指定T坐标轴贴图模式
	    gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_WRAP_T,GL10.GL_REPEAT);
	    
	    //声明输入流
	    InputStream is = context.getResources().openRawResource(textureId);
	    //创建Bitmap
	    Bitmap bitmapTmp; 
	    try 
	    {	
	    	//加载图片
	    	bitmapTmp = BitmapFactory.decodeStream(is);
	    }finally 
	    {
	        try 
	        {
	        	//关闭输入流
	            is.close();
	        } 
	        catch(IOException e) 
	        {
	            e.printStackTrace();
	        }
	    }
	    GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, bitmapTmp, 0);
	    //关闭bitmap
	    bitmapTmp.recycle(); 
	    
	    //返回生成的纹理ID
	    return currTextureId;
	}
	
	/**
	 * GET/SET 方法
	 */
	public Circle getCircle(){
		return circle;
	}
	
}
