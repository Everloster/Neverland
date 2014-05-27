package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.graphics.CircleSurfaceView;
import org.jabe.neverland.graphics.DataConfig;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.CompoundButton.OnCheckedChangeListener;
import android.widget.LinearLayout;
import android.widget.Toast;
import android.widget.ToggleButton;

public class CircleActivity extends Activity {
	private ToggleButton myToggleButton;
	
	private CircleSurfaceView circleSurfaceView;
	
	private Button myMagnifyButton;
	private Button myLessenButton;
	
	private Button myLeftButton;
	private Button myRightButton;
	private Button myUpButton;
	private Button myDownButton;
	
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.act_opengl_3d);
        
        //实例化视图
        circleSurfaceView = new CircleSurfaceView(this);
        //设置视图聚焦
        circleSurfaceView.requestFocus();
        //设置视图可触控
        circleSurfaceView.setFocusableInTouchMode(true);
        
        //获取当前布局
        LinearLayout myLinearLayout = (LinearLayout)findViewById(R.id.myLinearLayout);
        //将视图添加至布局中
        myLinearLayout.addView(circleSurfaceView);
        
        //获取ToggleButton
        myToggleButton = (ToggleButton)findViewById(R.id.myToggleButton);
        //设置ToggleButton监听事件
        myToggleButton.setOnCheckedChangeListener(new OnCheckedChangeListener() {
			@Override
			public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
				//修改打开灯光效果状态
				DataConfig.openLightFlag = !DataConfig.openLightFlag;
				System.out.println("当前灯效果状态："+DataConfig.openLightFlag);
			}
		});
        
        //获取放大按钮
        myMagnifyButton = (Button)findViewById(R.id.myMagnifyButton);
        //设置放大事件
        myMagnifyButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.scaleX>1.5){
					Toast.makeText(getApplicationContext(), "已放大到最大",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.scaleX += DataConfig.change;
					DataConfig.scaleY += DataConfig.change;
				}
				
			}
		});
        //获取缩小按钮
        myLessenButton = (Button)findViewById(R.id.myLessenButton);
        //设置缩小事件
        myLessenButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.scaleX<0.5){
					Toast.makeText(getApplicationContext(), "已缩小到最小",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.scaleX -= DataConfig.change;
					DataConfig.scaleY -= DataConfig.change;
				}
			}
		});
        //获取向左按钮
        myLeftButton = (Button)findViewById(R.id.myLeftButton);
        //设置向左事件
        myLeftButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.moveX<-3){
					Toast.makeText(getApplicationContext(), "已左移到最左边",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.moveX -= DataConfig.change;
					DataConfig.moveX -= DataConfig.change;
				}
			}
		});
        //获取向右按钮
        myRightButton = (Button)findViewById(R.id.myRightButton);
        //设置向右事件
        myRightButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.moveX>3){
					Toast.makeText(getApplicationContext(), "已右移到最右边",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.moveX += DataConfig.change;
					DataConfig.moveX += DataConfig.change;
				}
			}
		});
        //获取向上按钮
        myUpButton = (Button)findViewById(R.id.myUpButton);
        //设置向上事件
        myUpButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.moveY>3){
					Toast.makeText(getApplicationContext(), "已上移到最上边",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.moveY += DataConfig.change;
					DataConfig.moveY += DataConfig.change;
				}
			}
		});
        //获取向下按钮
        myDownButton = (Button)findViewById(R.id.myDownButton);
        //设置向下事件
        myDownButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if(DataConfig.moveY<-3){
					Toast.makeText(getApplicationContext(), "已下移到最下边",Toast.LENGTH_SHORT).show();
				}else{
					DataConfig.moveY -= DataConfig.change;
					DataConfig.moveY -= DataConfig.change;
				}
			}
		});
        
    }
    
    

	@Override
	protected void onPause() {
		// TODO Auto-generated method stub
		super.onPause();
		circleSurfaceView.onPause();
	}

	@Override
	protected void onResume() {
		// TODO Auto-generated method stub
		super.onResume();
		circleSurfaceView.onResume();
	}
}