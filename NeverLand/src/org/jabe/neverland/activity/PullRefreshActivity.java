package org.jabe.neverland.activity;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.jabe.neverland.R;
import org.jabe.neverland.model.CommonAdapter;
import org.jabe.neverland.model.ListElement;
import org.jabe.neverland.model.SimpleListItem;
import org.jabe.neverland.view.PullRefreshContainerView;
import org.jabe.neverland.view.PullRefreshContainerView.OnChangeStateListener;

import android.app.Activity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.ViewGroup.LayoutParams;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

public class PullRefreshActivity extends Activity {

	private ArrayAdapter<String> mAdapter;
	private PullRefreshContainerView mContainerView;
	private ListView mList;
	
	private TextView mRefreshHeader;
	private ArrayList<String> mStrings = new ArrayList<String>();
	LinkedList<String> data = new LinkedList<String>();
	CommonAdapter<String> baseAdapter;
	
    @SuppressWarnings("unused")
	private void addStrings(int count) {
    	int curSize = mStrings.size();
    	for(int i = 0; i < count; ++i) {
    		mStrings.add("String " + (curSize + i));
    	}
    	mAdapter.notifyDataSetChanged();
    }
    
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_pullrefresh_container);
		
		mRefreshHeader = new TextView(this);
        mRefreshHeader.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        mRefreshHeader.setGravity(Gravity.CENTER);
        mRefreshHeader.setText("Pull to refresh...");

        mContainerView = (PullRefreshContainerView) findViewById(R.id.container);
        mContainerView.setRefreshHeader(mRefreshHeader);
        
        mContainerView.setOnChangeStateListener(new OnChangeStateListener() {
    		@Override
    		public void onChangeState(PullRefreshContainerView container, int nowState, int oldState) {
    			switch(nowState) {
    			case PullRefreshContainerView.STATE_IDLE:
    			case PullRefreshContainerView.STATE_PULL:
    				mRefreshHeader.setText("Pull to refresh...");
    				break;
    			case PullRefreshContainerView.STATE_RELEASE:
    				mRefreshHeader.setText("Release to refresh...");
    				break;
    			case PullRefreshContainerView.STATE_LOADING:
    				mRefreshHeader.setText("Loading...");
    				new Thread(new Runnable() {
						@Override
						public void run() {
							PullRefreshActivity.this.runOnUiThread(new Runnable() {
								@Override
								public void run() {
									final int size = baseAdapter.getCount() + 1;
									baseAdapter.addFirst("string" + size);
									mContainerView.completeRefresh();
									baseAdapter.notifyDataSetChanged();
								}
		    				});
						}
    					
    				}).start();
    				break;
    			}
    		}
        });
        //normal
        mList = mContainerView.getList();
        mAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, mStrings);
        
        
        //common adapter
        List<ListElement<String>> allList = new LinkedList<ListElement<String>>();
        SimpleListItem tempListItem = new SimpleListItem(data, this);
		for (int i = 100; i > 0; i --) {
			data.addLast("string" + i);
			allList.add(tempListItem.clone());
		}
		baseAdapter = new CommonAdapter<String>(allList);
		
        mList.setAdapter(baseAdapter);
	}
	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		return super.onKeyDown(keyCode, event);
	}
}
